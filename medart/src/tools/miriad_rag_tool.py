"""
Miriad QA RAG search tool for LangChain agent.

单表 PubMed QA 知识库检索工具（MIRIAD，question/answer 对）：
- 输入：检索关键词（必选）→ 文本向量检索（question_vec 与 answer_vec 双路）
- 输出：question/answer 对 + 相似度，两路经 RRF 融合排序
- Milvus 单机服务器（http://127.0.0.1:19530），非 Lite。

数据集简介（摘自 MIRIAD-5.8M README）：
MIRIAD 是一个大规模精选医学指令与检索数据集（Medical Instruction and
RetrIeval Dataset），包含 580 万条医学问答对，由 LLM 从同行评审的生物医学
文献中蒸馏生成。MIRIAD 提供结构化、高质量的问-答对，可支撑 RAG、医学检索、
幻觉检测、指令微调等下游任务。数据集由 Open Data Commons Attribution
License (ODC-By) v1.0 授权发布，仅供学术研究与教育用途，未经监管机构审查，
不得用于临床决策或视为认证医疗器械。

表结构：本 tool 检索的 miriad_qa 是 MIRIAD 导入本地 Milvus 后的单个 collection。

加载数据集（复现/补导参考）：
    from datasets import load_dataset
    dataset = load_dataset("miriad/miriad-5.8M", split="train")

复用了 multimodal_rag_tool.py 的基础设施（避免重复实现）：
- PubMedBERT 文本编码器（懒加载单例、线程安全、CPU 推理）
- Milvus 客户端懒加载单例 + collection load 状态跟踪
- FIFO 公平门闸（多智能体并发调用时同一时刻只允许一个请求做编码/检索，
  防止并发把 GPU 显存/内存叠加打爆 OOM）
- 路径解析（RAG_MILVUS_URI / RAG_MODEL_ROOT 环境变量覆盖 + 多候选）

与 miriad_qa 表结构对应：
    id            INT64        主键
    question      VARCHAR      问题原文
    answer        VARCHAR      答案原文
    question_vec  FLOAT_VECTOR question 的 embedding (768)
    answer_vec    FLOAT_VECTOR answer  的 embedding (768)

检索策略：同一查询向量同时对 question_vec 和 answer_vec 各做一次 top-k 召回，
两路以 id 为 key 做 RRF 融合 —— question 命中表示"问题相似"，answer 命中表示
"答案相似"，融合后更稳。miriad_qa 两个向量字段同维度（768），不受 milvus-lite
混合维度多向量字段索引 bug 影响。

遵循项目 tool 约定（同 multimodal_rag_tool.py）：
- @tool + Pydantic args_schema
- 返回 (content_blocks, result) 元组：content_blocks 含文本块，多模态 LLM 可直接读；
  result 为应用层元数据
- 暴露纯函数 run_miriad_rag_search()，不依赖 langchain，可被流水线直接调用

注册到 agent：在 run_multi_agent.py 中
    from src.tools.miriad_rag_tool import retrieve_miriad_qa
    AVAILABLE_TOOLS = [..., retrieve_miriad_qa]
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

from src.tools.multimodal_rag_tool import (
    _RAG_GATE,
    _ensure_collection_loaded,
    _get_client,
    get_text_encoder,
)
# [EVIDENCE-FILTER] 实验性证据过滤(可整体移除): 详见 rag_evidence_filter.py
from src.tools.rag_evidence_filter import filter_rag_hits_by_evidence

logger = logging.getLogger("miriad_rag_tool")

# ---------------------------------------------------------------------------
# 表配置（与 milvus_import/import_miriad.py 保持一致）
# ---------------------------------------------------------------------------

COLLECTION_NAME = "miriad_qa"
QUESTION_VEC_FIELD = "question_vec"
ANSWER_VEC_FIELD = "answer_vec"
VEC_DIM = 768            # PubMedBERT base
QUESTION_FIELD = "question"
ANSWER_FIELD = "answer"
RRF_K = 60               # RRF 融合参数
CHANNEL_TOP_K = 10       # 每路检索召回数

# 双路检索: 同一个查询向量分别打 question_vec / answer_vec 两个字段
CHANNELS = [
    ("question", QUESTION_VEC_FIELD),
    ("answer", ANSWER_VEC_FIELD),
]


# ---------------------------------------------------------------------------
# 检索核心
# ---------------------------------------------------------------------------

def _search_field(query_vec, vec_field: str, top_k: int) -> list[dict[str, Any]]:
    """在 miriad_qa 的单个向量字段上做 top-k 检索, 归一化命中结果。"""
    client = _get_client()
    _ensure_collection_loaded(client, COLLECTION_NAME)
    res = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vec.tolist()],
        anns_field=vec_field,
        search_params={"metric_type": "COSINE", "params": {}},
        limit=top_k,
        output_fields=[QUESTION_FIELD, ANSWER_FIELD],
    )
    hits = []
    for hit in (res[0] if res else []):
        entity = hit.get("entity", {}) if isinstance(hit, dict) else hit.entity
        distance = hit.get("distance") if isinstance(hit, dict) else hit.distance
        pk = hit.get("id") if isinstance(hit, dict) else hit.id
        hits.append({
            "id": pk,
            "question": entity.get(QUESTION_FIELD, "") or "",
            "answer": entity.get(ANSWER_FIELD, "") or "",
            "similarity": float(distance) if distance is not None else None,
        })
    return hits


def _rrf_merge(question_hits: list[dict], answer_hits: list[dict],
               rrf_k: int = RRF_K) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion 融合两路召回, 同 id 合并。"""
    merged: dict[int, dict[str, Any]] = {}
    for channel, hits in (("question", question_hits), ("answer", answer_hits)):
        for rank, h in enumerate(hits):
            m = merged.setdefault(h["id"], {
                "id": h["id"],
                "question": h["question"],
                "answer": h["answer"],
                "rrf_score": 0.0,
                "similarity": {},
                "sources": [],
            })
            m["rrf_score"] += 1.0 / (rrf_k + rank + 1)
            m["similarity"][channel] = h["similarity"]
            m["sources"].append(channel)
            if not m["question"] and h["question"]:
                m["question"] = h["question"]
            if not m["answer"] and h["answer"]:
                m["answer"] = h["answer"]
    return sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)


# ---------------------------------------------------------------------------
# 纯函数入口（流水线可直接调用, 不依赖 langchain）
# ---------------------------------------------------------------------------

def run_miriad_rag_search(
    query: str,
    top_k: int = 5,
    encoder_key: str = "pubmedbert",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """单表双字段检索: 关键词 → question_vec + answer_vec 双路召回, RRF 融合后
    返回 (content_blocks, result)。content_blocks 供 LLM 阅读。"""

    def _error_return(msg: str, tb: Optional[str] = None):
        text = f"MIRIAD QA search failed: {msg}"
        if tb:
            text += f"\n\nFull traceback:\n{tb}"
        blocks = [{"type": "text", "text": text}]
        result = {
            "success": False, "collection": COLLECTION_NAME, "query": query,
            "total_hits": 0, "n_question_hits": 0, "n_answer_hits": 0,
            "hits": [], "warnings": [], "message": msg, "traceback": tb or "",
        }
        return blocks, result

    # 1. 输入校验
    query = (query or "").strip()
    if not query:
        return _error_return("query is empty; pass non-empty search keywords.")

    warnings: list[str] = []
    stage = "encoding query"
    try:
        # FIFO 门闸: 多智能体多线程并发调用时, 同一时刻只允许一个请求
        # 做编码+检索, 其余按到达顺序排队, 防止并发把显存/内存打爆 (OOM)。
        with _RAG_GATE:
            stage = f"text encoding ({encoder_key})"
            text_enc = get_text_encoder(encoder_key)
            q_vec = text_enc.encode_texts([query])[0]

            stage = f"vector search ({COLLECTION_NAME}.question_vec)"
            question_hits = _search_field(q_vec, QUESTION_VEC_FIELD, CHANNEL_TOP_K)

            stage = f"vector search ({COLLECTION_NAME}.answer_vec)"
            answer_hits = _search_field(q_vec, ANSWER_VEC_FIELD, CHANNEL_TOP_K)
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("MIRIAD QA retrieval failed at stage [%s]", stage)
        error_blocks, error_result = _error_return(
            f"retrieval error at [{stage}]: {type(e).__name__}: {e}", tb=tb)
        return error_blocks, error_result

    # 2. RRF 融合
    fused = _rrf_merge(question_hits, answer_hits)[:top_k]
    # [EVIDENCE-FILTER] 实验性: 仅保留支持查询的证据, 非证据命中不下发(可移除)
    fused = filter_rag_hits_by_evidence(query, fused, text_key="answer",
                                        collection=COLLECTION_NAME)

    # 3. content_blocks: 概览 + 逐条命中
    overview = [
        f"## MIRIAD QA Results (collection: {COLLECTION_NAME})\n",
        f"- Query: {query}",
        f"- Question-channel hits: {len(question_hits)} | "
        f"Answer-channel hits: {len(answer_hits)} | "
        f"Fused results: {len(fused)}\n",
    ]
    content_blocks: list[dict[str, Any]] = [
        {"type": "text", "text": "\n".join(overview)},
    ]
    for i, h in enumerate(fused, 1):
        sim = ", ".join(
            f"{ch}={h['similarity'].get(ch):.3f}"
            for ch in h["sources"] if h["similarity"].get(ch) is not None
        )
        hit_lines = [
            f"### Hit #{i}  [{'+'.join(h['sources'])} | sim: {sim} | "
            f"rrf: {h['rrf_score']:.4f}]",
            f"- question: {h['question'] or '(none)'}",
            f"- answer: {h['answer'] or '(none)'}",
        ]
        content_blocks.append({"type": "text", "text": "\n".join(hit_lines)})

    if warnings:
        content_blocks.append({
            "type": "text",
            "text": "**Warnings**:\n" + "\n".join(f"- {w}" for w in warnings),
        })

    # 4. result 元数据
    result: dict[str, Any] = {
        "success": True,
        "collection": COLLECTION_NAME,
        "query": query,
        "total_hits": len(fused),
        "n_question_hits": len(question_hits),
        "n_answer_hits": len(answer_hits),
        "hits": [
            {
                "rank": i,
                "id": h["id"],
                "question": h["question"],
                "answer": h["answer"],
                "sources": h["sources"],
                "similarity": h["similarity"],
                "rrf_score": round(h["rrf_score"], 6),
            }
            for i, h in enumerate(fused, 1)
        ],
        "warnings": warnings,
        "message": (
            f"Retrieved {len(fused)} result(s) from collection '{COLLECTION_NAME}'."
        ),
    }
    return content_blocks, result


# ---------------------------------------------------------------------------
# LangChain tool
# ---------------------------------------------------------------------------

class MiriadRagSearchInput(BaseModel):
    query: str = Field(
        description=(
            "Required. Search keywords describing the medical question or "
            "answer of interest, e.g. 'pulmonary embolism treatment' or "
            "'what causes chest pain'. Used for text-vector retrieval against "
            "the MIRIAD PubMed QA knowledge base."
        ),
        min_length=1,
    )
    top_k: Optional[int] = Field(
        default=5,
        description="Optional. Max number of fused results to return (default 5).",
    )


@tool(args_schema=MiriadRagSearchInput)
def retrieve_miriad_qa(
    query: str,
    top_k: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """从 MIRIAD 医学问答知识库中检索参考资料（问题-答案对）。
    MIRIAD is a large-scale curated medical QA corpus: 5.8M question-answer
    pairs, each rephrased from and grounded in a passage of peer-reviewed
    medical literature, spanning 56 medical disciplines. Unlike
    unstructured-text RAG, its query-response format enables targeted
    retrieval and demonstrably improves LLM accuracy on medical QA benchmarks
    and hallucination detection.

    The query keywords are embedded and matched against BOTH stored questions
    and stored answers; the two result lists are fused (RRF) and the top hits
    are returned as question/answer pairs.

    Useful when you need: a canonical medical answer, related follow-up
    questions, or supporting literature-derived context for a claim. Matched
    answers reflect the source literature, not a clinical recommendation.

    Inputs:
    - query: Search keywords for the medical question/answer (required)
    - top_k: Optional max fused results (default 5)

    Outputs (content_blocks → visible to LLM):
    - Text: Overview + ranked hits, each with question, answer, source
      channel(s) (question/answer), and similarity scores

    Outputs (result dict → application metadata):
    - success / total_hits / n_question_hits / n_answer_hits
    - hits: [{rank, id, question, answer, sources, similarity, rrf_score}]
    - warnings: Non-fatal issues
    - message: Status description
    """
    return run_miriad_rag_search(query=query, top_k=top_k or 5)


# 兼容命名: 与现有 multimodal_rag_tool 的 preload 语义对齐, 供启动时预加载编码器
def preload_miriad_models(encoder_key: str = "pubmedbert") -> None:
    """启动前预加载本工具用到的文本编码器并常驻, 避免首次调用现场加载。"""
    get_text_encoder(encoder_key)
    logger.info("MIRIAD RAG encoder preloaded: %s", encoder_key)