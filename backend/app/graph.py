from __future__ import annotations

import json
import uuid
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from .history import HistoryStore
from .index import HybridIndex
from .providers import get_chat_model
from .schemas import ChatResponse, Source
from .settings import Settings


MAX_AGENTIC_QUERIES = 3
MAX_AGENTIC_ATTEMPTS = 2
MIN_CONTEXT_CHARS_FOR_JUDGEMENT = 8


class RagState(TypedDict, total=False):
    session_id: str
    question: str
    rewritten_question: str
    history: list[dict[str, str]]
    scenario: str | None
    model: str | None
    top_k: int
    agentic: bool
    retrieval_plan: dict[str, Any]
    retrieval_attempts: int
    context_filter: dict[str, Any]
    context_judgement: dict[str, Any]
    verification: dict[str, Any]
    sources: list[Source]
    retrieval_debug: dict[str, Any]
    answer: str
    direct_answer: str


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
        agentic: bool = False,
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
                "agentic": agentic,
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
        graph.add_node("plan", self._plan_retrieval)
        graph.add_node("direct_answer", self._direct_answer)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("agentic_retrieve", self._retrieve_agentic)
        graph.add_node("filter_context", self._filter_context)
        graph.add_node("judge_context", self._judge_context)
        graph.add_node("generate", self._generate)
        graph.add_node("verify_answer", self._verify_answer)
        graph.set_entry_point("rewrite")
        graph.add_conditional_edges(
            "rewrite",
            self._route_after_rewrite,
            {"standard": "retrieve", "agentic": "plan"},
        )
        graph.add_edge("retrieve", "generate")
        graph.add_conditional_edges(
            "plan",
            self._route_after_plan,
            {"retrieve": "agentic_retrieve", "direct": "direct_answer"},
        )
        graph.add_edge("direct_answer", END)
        graph.add_edge("agentic_retrieve", "filter_context")
        graph.add_edge("filter_context", "judge_context")
        graph.add_conditional_edges(
            "judge_context",
            self._route_after_context_judgement,
            {"retry": "agentic_retrieve", "generate": "generate"},
        )
        graph.add_conditional_edges(
            "generate",
            self._route_after_generate,
            {"verify": "verify_answer", "end": END},
        )
        graph.add_edge("verify_answer", END)
        return graph.compile()

    def _route_after_rewrite(self, state: RagState) -> str:
        return "agentic" if state.get("agentic") else "standard"

    def _route_after_plan(self, state: RagState) -> str:
        plan = state.get("retrieval_plan") or {}
        return "retrieve" if _coerce_bool(plan.get("needs_retrieval"), default=True) else "direct"

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

    def _plan_retrieval(self, state: RagState) -> RagState:
        fallback_query = state.get("rewritten_question") or state["question"]
        fallback_subqueries = [{"id": "q1", "question": fallback_query, "purpose": "fallback"}]
        fallback_plan = {
            "needs_retrieval": True,
            "intent": "fallback",
            "queries": [fallback_query],
            "subqueries": fallback_subqueries,
            "retry_strategy": "same_query",
            "notes": "planner unavailable or returned no usable queries",
        }
        data = self._invoke_agentic_json(
            state,
            system=(
                "You are a retrieval planner for an enterprise RAG system. "
                "Return only JSON with keys: needs_retrieval, intent, subqueries, "
                "retry_strategy, direct_answer, notes. "
                "Use at most three subqueries. Each subquery has id, question, purpose. "
                "Set needs_retrieval=false only for greetings, capability questions, or clarification requests."
            ),
            user=(
                f"Question: {state['question']}\n"
                f"Rewritten question: {fallback_query}\n"
                f"Scenario filter: {state.get('scenario') or 'none'}"
            ),
            max_tokens=400,
        )
        plan = fallback_plan
        direct_answer = ""
        if data:
            needs_retrieval = _coerce_bool(data.get("needs_retrieval"), default=True)
            subqueries = _normalize_subqueries(data, fallback_query)
            queries = _queries_from_subqueries(subqueries)
            if queries:
                plan = {
                    "needs_retrieval": needs_retrieval,
                    "intent": str(data.get("intent") or "general"),
                    "queries": queries,
                    "subqueries": subqueries,
                    "retry_strategy": str(data.get("retry_strategy") or "query_rewrite"),
                    "notes": str(data.get("notes") or ""),
                }
                direct_answer = str(data.get("direct_answer") or "")
        debug = _with_agentic_debug(state.get("retrieval_debug"), plan=plan)
        return {
            "retrieval_plan": plan,
            "retrieval_attempts": 0,
            "retrieval_debug": debug,
            "direct_answer": direct_answer,
        }

    def _direct_answer(self, state: RagState) -> RagState:
        answer = (state.get("direct_answer") or "").strip()
        if not answer:
            answer = "请补充一个需要查询知识库的具体问题。"
        debug = _with_agentic_debug(state.get("retrieval_debug"), direct_answer=True)
        return {"answer": answer, "sources": [], "retrieval_debug": debug}

    def _retrieve_agentic(self, state: RagState) -> RagState:
        attempt = int(state.get("retrieval_attempts") or 0) + 1
        fallback_query = state.get("rewritten_question") or state["question"]
        if attempt > 1:
            judgement = state.get("context_judgement") or {}
            raw_queries = judgement.get("retry_queries")
            retry_strategy = str(judgement.get("retry_strategy") or "query_rewrite")
        else:
            raw_queries = (state.get("retrieval_plan") or {}).get("queries")
            retry_strategy = str((state.get("retrieval_plan") or {}).get("retry_strategy") or "initial")
        attempted_queries = _attempted_queries(state.get("retrieval_debug"))
        base_queries = _normalize_queries(raw_queries, fallback_query)
        if attempt == 1:
            base_queries = _merge_unique_strings([fallback_query], base_queries)[:MAX_AGENTIC_QUERIES]
        queries = _exclude_attempted_queries(base_queries, attempted_queries)
        if not queries:
            queries = base_queries
        merged_sources: list[Source] = []
        seen_chunk_ids: set[str] = set()
        retrievals: list[dict[str, Any]] = []
        final_top_k = state.get("top_k") or self.settings.final_top_k
        if attempt > 1:
            for source in state.get("sources", []):
                if source.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(source.chunk_id)
                merged_sources.append(source)
                if len(merged_sources) >= final_top_k:
                    break
        for query in queries:
            sources, debug = self.index.search(
                query,
                scenario=state.get("scenario"),
                dense_top_k=self.settings.dense_top_k,
                bm25_top_k=self.settings.bm25_top_k,
                final_top_k=final_top_k,
                rrf_k=self.settings.rrf_k,
            )
            retrievals.append({"query": query, "debug": debug, "source_count": len(sources)})
            for source in sources:
                if source.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(source.chunk_id)
                merged_sources.append(source)
                if len(merged_sources) >= final_top_k:
                    break
            if len(merged_sources) >= final_top_k:
                break
        agentic_debug = {
            "attempt": attempt,
            "queries": queries,
            "retry_strategy": retry_strategy,
            "retrievals": retrievals,
            "source_count": len(merged_sources),
        }
        debug = _with_agentic_attempt(state.get("retrieval_debug"), agentic_debug)
        debug["retrieval_backend"] = "agentic_milvus_hybrid"
        debug["index_operation"] = "search_only"
        return {"sources": merged_sources, "retrieval_attempts": attempt, "retrieval_debug": debug}

    def _filter_context(self, state: RagState) -> RagState:
        sources = state.get("sources", [])
        if not sources:
            context_filter = {
                "kept_documents": [],
                "rejected_documents": [],
                "kept_count": 0,
                "rejected_count": 0,
                "reason": "no retrieved sources",
            }
            debug = _with_agentic_debug(state.get("retrieval_debug"), context_filter=context_filter)
            return {"sources": [], "context_filter": context_filter, "retrieval_debug": debug}
        data = self._invoke_agentic_json(
            state,
            system=(
                "You filter retrieved documents for answer relevance. "
                "Return only JSON with keys: kept_documents, rejected_documents, reason. "
                "Use document refs like D1, source_path, or source_name. "
                "Keep a document if any chunk in it directly helps answer the question. "
                "Reject only whole documents that are unrelated."
            ),
            user=(
                f"Question: {state['question']}\n"
                f"Retrieved documents:\n{_document_snippets(sources)}"
            ),
            max_tokens=500,
        )
        if not data:
            kept_sources = sources
            rejected_document_keys: list[str] = []
            reason = "filter unavailable; kept all retrieved sources"
        else:
            document_keys = _document_keys(sources)
            kept_document_keys = _normalize_document_references(data.get("kept_documents"), sources)
            kept_document_keys = _merge_unique_strings(
                kept_document_keys,
                _document_keys_from_chunk_refs(data.get("kept_chunk_ids"), sources),
            )
            rejected_document_keys = [
                document_key
                for document_key in _normalize_document_references(data.get("rejected_documents"), sources)
                if document_key in document_keys and document_key not in kept_document_keys
            ]
            rejected_document_keys = _merge_unique_strings(
                rejected_document_keys,
                [
                    document_key
                    for document_key in _document_keys_from_chunk_refs(data.get("rejected_chunk_ids"), sources)
                    if document_key not in kept_document_keys
                ],
            )
            if kept_document_keys:
                kept_sources = [source for source in sources if _document_key(source) in kept_document_keys]
            elif rejected_document_keys and len(set(rejected_document_keys)) == len(document_keys):
                kept_sources = []
            else:
                kept_sources = sources
                rejected_document_keys = []
            reason = str(data.get("reason") or "")
        kept_ids = [source.chunk_id for source in kept_sources]
        kept_document_keys = _document_keys(kept_sources)
        rejected_sources = [source for source in sources if _document_key(source) in rejected_document_keys]
        context_filter = {
            "kept_documents": kept_document_keys,
            "rejected_documents": rejected_document_keys,
            "kept_count": len(kept_ids),
            "rejected_count": len(rejected_sources),
            "reason": reason,
        }
        debug = _with_agentic_debug(state.get("retrieval_debug"), context_filter=context_filter)
        return {"sources": kept_sources, "context_filter": context_filter, "retrieval_debug": debug}

    def _judge_context(self, state: RagState) -> RagState:
        sources = state.get("sources", [])
        fallback_query = state.get("rewritten_question") or state["question"]
        attempts = int(state.get("retrieval_attempts") or 0)
        can_retry = attempts < MAX_AGENTIC_ATTEMPTS
        context_filter = state.get("context_filter") or {}
        attempted_queries = _attempted_queries(state.get("retrieval_debug"))
        if not sources:
            low_relevance = int(context_filter.get("rejected_count") or 0) > 0
            strategy = "keyword" if low_relevance else "broaden"
            retry_queries = _build_retry_queries(state, strategy, fallback_query, attempted_queries) if can_retry else []
            judgement = {
                "sufficient": False,
                "missing": str(context_filter.get("reason") or "no retrieved sources"),
                "retry_strategy": strategy if can_retry else "none",
                "retry_queries": retry_queries,
                "retry_query_source": "rule_builder" if retry_queries else "none",
                "attempted_queries": attempted_queries,
                "action": "retry" if can_retry else "evidence_limited_answer",
            }
        elif _is_vague_question(state["question"]):
            judgement = {
                "sufficient": False,
                "missing": "question is ambiguous",
                "retry_strategy": "none",
                "retry_queries": [],
                "retry_query_source": "none",
                "attempted_queries": attempted_queries,
                "action": "clarify",
                "clarifying_question": "请补充你指的是哪个具体对象、流程、制度或问题场景。",
            }
        elif _context_is_too_short(sources):
            retry_queries = _build_retry_queries(state, "query_rewrite", fallback_query, attempted_queries) if can_retry else []
            judgement = {
                "sufficient": False,
                "missing": "retrieved sources are too short to answer reliably",
                "retry_strategy": "query_rewrite" if can_retry else "none",
                "retry_queries": retry_queries,
                "retry_query_source": "rule_builder" if retry_queries else "none",
                "attempted_queries": attempted_queries,
                "action": "retry" if can_retry else "evidence_limited_answer",
            }
        else:
            data = self._invoke_agentic_json(
                state,
                system=(
                    "You judge whether retrieved context is enough to answer. "
                    "Return only JSON with keys: sufficient, missing, retry_strategy, retry_queries. "
                    "retry_strategy must be one of same_query, broaden, narrow, entity_expand, keyword, query_rewrite."
                ),
                user=(
                    f"Question: {state['question']}\n"
                    f"Retrieved context:\n{_source_snippets(sources)}"
                ),
                max_tokens=300,
            )
            if data:
                sufficient = _coerce_bool(data.get("sufficient"), default=True)
                retry_queries = (
                    _exclude_attempted_queries(
                        _normalize_queries(data.get("retry_queries"), fallback_query),
                        attempted_queries,
                    )
                    if not sufficient and can_retry
                    else []
                )
                judgement = {
                    "sufficient": sufficient,
                    "missing": str(data.get("missing") or ""),
                    "retry_strategy": str(data.get("retry_strategy") or "query_rewrite")
                    if not sufficient and can_retry
                    else "none",
                    "retry_queries": retry_queries,
                    "retry_query_source": "judge_model" if retry_queries else "none",
                    "attempted_queries": attempted_queries,
                    "action": _judgement_action(sufficient=sufficient, can_retry=can_retry, retry_queries=retry_queries),
                }
            else:
                judgement = {
                    "sufficient": True,
                    "missing": "",
                    "retry_strategy": "none",
                    "retry_queries": [],
                    "retry_query_source": "none",
                    "attempted_queries": attempted_queries,
                    "action": "generate",
                }
        debug = _with_agentic_debug(state.get("retrieval_debug"), context_judgement=judgement)
        return {"context_judgement": judgement, "retrieval_debug": debug}

    def _route_after_context_judgement(self, state: RagState) -> str:
        judgement = state.get("context_judgement") or {}
        attempts = int(state.get("retrieval_attempts") or 0)
        if (
            not _coerce_bool(judgement.get("sufficient"), default=True)
            and attempts < MAX_AGENTIC_ATTEMPTS
            and judgement.get("retry_queries")
        ):
            return "retry"
        return "generate"

    def _route_after_generate(self, state: RagState) -> str:
        return "verify" if state.get("agentic") else "end"

    def _generate(self, state: RagState) -> RagState:
        sources = state.get("sources", [])
        context = format_context(sources)
        system = (
            "你是一个严谨的 RAG 问答助手。"
            "你的任务是根据提供的引用资料回答用户问题。"
            "回答必须忠实于引用资料，不得编造、猜测或使用资料外的信息。"
            "如果资料不足，请明确说明“根据当前资料无法确定”，并说明还需要哪些信息。"
            "如果资料之间存在冲突，请指出冲突。"
            "关键事实后必须标注引用编号。"
            "回答应简洁、清晰，控制在 300 个汉字以内。"
        )
        user = (
            f"用户问题: {state['question']}\n\n"
            f"检索问题: {state.get('rewritten_question') or state['question']}\n\n"
            f"引用资料:\n{context}\n\n"
            "流程类问题说明步骤；"
            "规则类问题说明条件、限制和例外；"
            "事实类问题直接给出结论；"
        )
        user += "\nAnswer in the same language as the user's question."
        llm = get_chat_model(
            self.settings,
            state.get("model"),
            temperature=0.2,
            thinking=False,
            max_tokens=700,
        )
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return {"answer": str(response.content).strip()}

    def _verify_answer(self, state: RagState) -> RagState:
        answer = state.get("answer", "")
        sources = state.get("sources", [])
        if not sources:
            verification = {
                "supported": False,
                "reason": "no retrieved sources",
            }
            repaired = _unsupported_answer(state)
            debug = _with_agentic_debug(state.get("retrieval_debug"), verification=verification)
            return {"answer": repaired, "verification": verification, "retrieval_debug": debug}
        data = self._invoke_agentic_json(
            state,
            system=(
                "You verify whether an answer is fully supported by retrieved sources. "
                "Return only JSON with keys: supported, reason."
            ),
            user=(
                f"Question: {state['question']}\n"
                f"Answer: {answer}\n"
                f"Retrieved context:\n{_source_snippets(sources)}"
            ),
            max_tokens=300,
        )
        supported = _coerce_bool(data.get("supported"), default=True) if data else True
        verification = {
            "supported": supported,
            "reason": str(data.get("reason") or "") if data else "",
        }
        result: RagState = {"verification": verification}
        result["retrieval_debug"] = _with_agentic_debug(state.get("retrieval_debug"), verification=verification)
        return result

    def _invoke_agentic_json(
        self,
        state: RagState,
        *,
        system: str,
        user: str,
        max_tokens: int,
    ) -> dict[str, Any] | None:
        try:
            llm = get_chat_model(
                self.settings,
                state.get("model"),
                temperature=0.0,
                thinking=False,
                max_tokens=max_tokens,
            )
            response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        except Exception:
            return None
        return _parse_json_object(str(response.content))


def format_context(sources: list[Source]) -> str:
    blocks: list[str] = []
    for index, source in enumerate(sources, start=1):
        page = f", 页码: {source.page}" if source.page is not None else ""
        section = f", 章节: {source.section}" if source.section else ""
        blocks.append(
            f"[{index}] 来源: {source.source_name}{page}{section}, 场景: {source.scenario}\n{source.text}"
        )
    return "\n\n".join(blocks) if blocks else "无可用引用。"


def _normalize_queries(raw_queries: Any, fallback_query: str) -> list[str]:
    if isinstance(raw_queries, str):
        candidates = [raw_queries]
    elif isinstance(raw_queries, list):
        candidates = raw_queries
    else:
        candidates = []
    queries: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        query = candidate.strip()
        if not query or query in seen:
            continue
        seen.add(query)
        queries.append(query)
        if len(queries) >= MAX_AGENTIC_QUERIES:
            break
    return queries or [fallback_query]


def _normalize_subqueries(data: dict[str, Any], fallback_query: str) -> list[dict[str, str]]:
    raw_subqueries = data.get("subqueries")
    subqueries: list[dict[str, str]] = []
    seen: set[str] = set()
    if isinstance(raw_subqueries, list):
        for index, item in enumerate(raw_subqueries, start=1):
            if isinstance(item, dict):
                question = str(item.get("question") or "").strip()
                purpose = str(item.get("purpose") or "retrieve evidence").strip()
                subquery_id = str(item.get("id") or f"q{index}").strip()
            elif isinstance(item, str):
                question = item.strip()
                purpose = "retrieve evidence"
                subquery_id = f"q{index}"
            else:
                continue
            if not question or question in seen:
                continue
            seen.add(question)
            subqueries.append({"id": subquery_id or f"q{index}", "question": question, "purpose": purpose})
            if len(subqueries) >= MAX_AGENTIC_QUERIES:
                break
    if subqueries:
        return subqueries
    return [
        {"id": f"q{index}", "question": query, "purpose": "retrieve evidence"}
        for index, query in enumerate(_normalize_queries(data.get("queries"), fallback_query), start=1)
    ]


def _queries_from_subqueries(subqueries: list[dict[str, str]]) -> list[str]:
    return [item["question"] for item in subqueries if item.get("question")]


def _normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
    return result


def _parse_json_object(content: str) -> dict[str, Any] | None:
    text = content.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _source_snippets(sources: list[Source], limit: int = 4) -> str:
    snippets = []
    for index, source in enumerate(sources[:limit], start=1):
        snippets.append(f"[{index}] chunk_id={source.chunk_id} source={source.source_name}: {source.text[:600]}")
    return "\n\n".join(snippets)


def _document_snippets(sources: list[Source]) -> str:
    documents: dict[str, list[Source]] = {}
    for source in sources:
        documents.setdefault(_document_key(source), []).append(source)
    blocks: list[str] = []
    for index, (document_key, document_sources) in enumerate(documents.items(), start=1):
        source_name = document_sources[0].source_name
        chunks = "\n".join(
            f"- chunk_id={source.chunk_id}: {source.text[:400]}" for source in document_sources[:4]
        )
        blocks.append(f"[D{index}] source_path={document_key} source_name={source_name}\n{chunks}")
    return "\n\n".join(blocks)


def _document_key(source: Source) -> str:
    return source.source_path or source.source_name


def _document_keys(sources: list[Source]) -> list[str]:
    return _merge_unique_strings([], [_document_key(source) for source in sources])


def _normalize_document_references(value: Any, sources: list[Source]) -> list[str]:
    document_keys = _document_keys(sources)
    by_reference: dict[str, str] = {}
    for index, document_key in enumerate(document_keys, start=1):
        source_name = next((source.source_name for source in sources if _document_key(source) == document_key), "")
        for reference in (document_key, source_name, str(index), f"D{index}", f"d{index}"):
            if reference:
                by_reference[reference.strip().lower()] = document_key
    result: list[str] = []
    seen: set[str] = set()
    for item in _normalize_string_list(value):
        document_key = by_reference.get(item.strip().lower())
        if not document_key or document_key in seen:
            continue
        seen.add(document_key)
        result.append(document_key)
    return result


def _document_keys_from_chunk_refs(value: Any, sources: list[Source]) -> list[str]:
    by_chunk_id = {source.chunk_id: _document_key(source) for source in sources}
    return _merge_unique_strings(
        [],
        [by_chunk_id[chunk_id] for chunk_id in _normalize_source_references(value, sources) if chunk_id in by_chunk_id],
    )


def _normalize_source_references(value: Any, sources: list[Source]) -> list[str]:
    source_ids = {source.chunk_id for source in sources}
    ordinal_ids = {str(index): source.chunk_id for index, source in enumerate(sources, start=1)}
    result: list[str] = []
    seen: set[str] = set()
    for item in _normalize_string_list(value):
        chunk_id = item if item in source_ids else ordinal_ids.get(item)
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        result.append(chunk_id)
    return result


def _with_agentic_debug(debug: dict[str, Any] | None, **updates: Any) -> dict[str, Any]:
    next_debug = dict(debug or {})
    agentic = dict(next_debug.get("agentic") or {})
    agentic.update(updates)
    next_debug["agentic"] = agentic
    return next_debug


def _with_agentic_attempt(debug: dict[str, Any] | None, attempt_debug: dict[str, Any]) -> dict[str, Any]:
    next_debug = _with_agentic_debug(debug, **attempt_debug)
    agentic = dict(next_debug.get("agentic") or {})
    attempt_number = attempt_debug.get("attempt")
    attempts = [
        item
        for item in list(agentic.get("attempts") or [])
        if not isinstance(item, dict) or item.get("attempt") != attempt_number
    ]
    attempts.append(dict(attempt_debug))
    agentic["attempts"] = attempts
    agentic["attempted_queries"] = _merge_unique_strings(_attempted_queries({"agentic": agentic}), attempt_debug.get("queries"))
    next_debug["agentic"] = agentic
    return next_debug


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    if isinstance(value, int | float):
        return bool(value)
    return default


def _context_is_too_short(sources: list[Source]) -> bool:
    total_chars = sum(len(source.text.strip()) for source in sources)
    return total_chars < MIN_CONTEXT_CHARS_FOR_JUDGEMENT


def _is_vague_question(question: str) -> bool:
    text = question.strip().lower()
    if not text:
        return True
    vague_phrases = (
        "这个怎么处理",
        "那个怎么处理",
        "它怎么处理",
        "这个怎么办",
        "那个怎么办",
        "它怎么办",
        "有什么要求",
        "怎么走",
        "怎么弄",
        "怎么处理",
    )
    if any(phrase in text for phrase in vague_phrases):
        anchors = ("接口", "500", "流程", "制度", "审批", "报销", "权限", "变更", "发布", "故障")
        return not any(anchor in text for anchor in anchors)
    return len(text) <= 4 and text in {"怎么办", "怎么做", "要求", "流程", "处理"}


def _judgement_action(*, sufficient: bool, can_retry: bool, retry_queries: list[str]) -> str:
    if sufficient:
        return "generate"
    if can_retry and retry_queries:
        return "retry"
    return "evidence_limited_answer"


def _attempted_queries(debug: dict[str, Any] | None) -> list[str]:
    if not isinstance(debug, dict):
        return []
    agentic = debug.get("agentic")
    if not isinstance(agentic, dict):
        return []
    queries = _normalize_string_list(agentic.get("attempted_queries"))
    attempts = agentic.get("attempts")
    if isinstance(attempts, list):
        for attempt in attempts:
            if isinstance(attempt, dict):
                queries = _merge_unique_strings(queries, attempt.get("queries"))
    return queries


def _build_retry_queries(
    state: RagState,
    strategy: str,
    fallback_query: str,
    attempted_queries: list[str],
) -> list[str]:
    question = state["question"]
    rewritten = state.get("rewritten_question") or fallback_query
    if strategy == "broaden":
        candidates = _broaden_queries(rewritten, question)
    elif strategy == "query_rewrite":
        candidates = [
            rewritten,
            f"{question} process conditions steps requirements",
            f"{rewritten} process conditions steps requirements",
        ]
    elif strategy in {"keyword", "entity_expand"}:
        keywords = _extract_query_keywords(rewritten or question)
        candidates = _keyword_retry_queries(keywords)
    else:
        candidates = [fallback_query]
    return _exclude_attempted_queries(_merge_unique_strings([], candidates), attempted_queries)[:MAX_AGENTIC_QUERIES]


def _broaden_queries(rewritten: str, question: str) -> list[str]:
    candidates = []
    for query in (rewritten, question):
        simplified = _remove_weak_query_terms(query)
        candidates.append(simplified)
        words = simplified.split()
        if len(words) > 3:
            candidates.append(" ".join(words[:3]))
        if len(words) > 2:
            candidates.append(" ".join(words[:2]))
    return candidates


def _keyword_retry_queries(keywords: list[str]) -> list[str]:
    if not keywords:
        return []
    candidates = [" ".join(keywords[:4])]
    if len(keywords) >= 3:
        candidates.append(" ".join([keywords[0], keywords[1], keywords[-1]]))
    if len(keywords) >= 2:
        candidates.append(" ".join(keywords[:2] + ["process", "requirements"]))
    return candidates


def _extract_query_keywords(query: str) -> list[str]:
    tokens = []
    for raw_token in query.replace("?", " ").replace("？", " ").replace(",", " ").replace("，", " ").split():
        token = raw_token.strip().strip(".:;；：()（）[]【】")
        if len(token) < 2:
            continue
        if token.lower() in {"what", "when", "where", "which", "how", "does", "the", "and", "for"}:
            continue
        tokens.append(token)
    return _merge_unique_strings([], tokens)


def _remove_weak_query_terms(query: str) -> str:
    result = query
    weak_terms = (
        "exact",
        "specific",
        "detailed",
        "current",
        "first step",
        "具体",
        "详细",
        "当前",
        "第一步",
        "这个",
        "那个",
    )
    for term in weak_terms:
        result = result.replace(term, " ")
    return " ".join(result.split()).strip() or query


def _exclude_attempted_queries(candidates: list[str], attempted_queries: list[str]) -> list[str]:
    attempted = {query.strip().lower() for query in attempted_queries if query.strip()}
    return [query for query in candidates if query.strip().lower() not in attempted]


def _merge_unique_strings(existing: list[str], values: Any) -> list[str]:
    result = list(existing)
    seen = {item.strip().lower() for item in result}
    for item in _normalize_string_list(values):
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _prefers_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _unsupported_answer(state: RagState) -> str:
    judgement = state.get("context_judgement") or {}
    missing = str(judgement.get("missing") or "").strip()
    if not _prefers_chinese(state.get("question", "")):
        if missing:
            return f"Unable to determine from the current sources. Additional information needed: {missing}"
        return "Unable to determine from the current sources. Add relevant source material before answering."
    if missing:
        return f"根据当前资料无法确定。还需要补充：{missing}"
    return "根据当前资料无法确定。还需要补充相关资料后再回答。"
