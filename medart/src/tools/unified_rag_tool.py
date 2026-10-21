"""
统一 RAG 检索工具（合并入口, 仲裁式路由）。

把三个独立 RAG 工具合并成一个对外入口:
- retrieve_medical_knowledge    (multimodal_rag_tool.py,   MedPix-2.0 影像+描述)
- retrieve_radimagenet_knowledge (radimagenet_rag_tool.py,  RadImageNet-VQA QA+影像)
- retrieve_miriad_qa            (miriad_rag_tool.py,       MIRIAD 医学问答纯文本)

本文件【不】改动任何旧文件 —— 它们保持原样, 由本文件依次全部调用, 对各自返回
结果做统一打分, 仲裁出得分最高的知识库, 只把那一家的 content_blocks 交给 LLM。

打分标准 (跨库可比):
- 三个库的文本通道都走同一个 PubMedBERT、图像通道都走同一个 MedImageInsight,
  余弦相似度 (similarity) 是同一尺度, 可以横向比较。
- 【不能拿来比】的是各家内部的 rrf_score —— 它是排名分 (1/(60+rank)), 只看
  排序位置不看匹配质量, 跨库无意义。
- 故每家得分 = top-3 命中"文本通道相似度"的均值 + 0.3 × "图像通道相似度"均值
  (若该库跑了图像通道)。一个命中都没有 → 0 分。
- 旧工具失败时返回 success=False 的结果, 本文件捕获后记为不可用,
  不参与仲裁 (也不抛异常)。

注册到 agent: 在 run_multi_agent.py 中
    from src.tools.unified_rag_tool import retrieve_unified_knowledge
    AVAILABLE_TOOLS = [..., retrieve_unified_knowledge]  # 可替换/保留旧三个工具

遵循项目 tool 约定 (同 multimodal_rag_tool.py):
- @tool + Pydantic args_schema
- 返回 (content_blocks, result) 元组
- 暴露纯函数 run_unified_rag_search(), 不依赖 langchain, 可被流水线直接调用
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import threading
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

# 调用各工具的纯函数入口(非 @tool 包装的 StructuredTool)。
# 旧文件的 @tool 是结构化工具, 不能当普通函数直接调(会得到
# "'StructuredTool' object is not callable")。
from src.tools.miriad_rag_tool import run_miriad_rag_search
from src.tools.multimodal_rag_tool import run_multimodal_rag_search
from src.tools.radimagenet_rag_tool import run_radimagenet_rag_search

logger = logging.getLogger("unified_rag_tool")

# ---------------------------------------------------------------------------
# 调用记录: retrieve_unified_knowledge 被哪个题目(样本)调用
# ---------------------------------------------------------------------------

# 当前样本上下文(题目), 由 run_multi_agent 在进入 agent 前注入 contextvar,
# 工具层读取后随调用记录落盘; 多线程各样本互不干扰 (与 rag_evidence_filter
# 的 set_claims_for_filter 同一套模式)。
_CURRENT_CASE: ContextVar[dict[str, Any]] = ContextVar(
    "unified_rag_current_case", default={}
)


def set_current_case_context(
    sample_id: Any = "unknown",
    claim: str = "",
    claim_type: str = "unknown",
) -> None:
    """[CALL-LOG] 记录当前处理样本的上下文, 供 retrieve_unified_knowledge 落盘。"""
    _CURRENT_CASE.set({
        "sample_id": str(sample_id),
        "claim": claim or "",
        "claim_type": claim_type or "unknown",
    })


def _current_case() -> dict[str, Any]:
    return _CURRENT_CASE.get() or {}


# 调用记录 JSONL 落盘路径: 默认项目根目录, 可用 set_rag_call_log_file() 指到输出目录
RAG_CALL_LOG_FILE = str(
    Path(__file__).resolve().parents[2] / "retrieve_unified_calls.jsonl"
)
_RAG_CALL_LOG_LOCK = threading.Lock()


def set_rag_call_log_file(path: str | os.PathLike) -> None:
    """[CALL-LOG] 重定向 retrieve_unified_knowledge 调用记录文件路径。"""
    global RAG_CALL_LOG_FILE
    RAG_CALL_LOG_FILE = str(path)


def _log_rag_call(payload: dict[str, Any]) -> None:
    """线程安全地把一条调用记录以 JSONL 追加写入; 失败只记日志, 不影响检索。"""
    try:
        with _RAG_CALL_LOG_LOCK:
            with open(RAG_CALL_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        logger.exception("Failed to write unified RAG call log.")

# 顺序: 先图像库, 后纯文本库 (打分仲裁与顺序无关, 仅决定 result 字段排列)
_SOURCES = [
    ("medpix2", run_multimodal_rag_search),
    ("radimagenet", run_radimagenet_rag_search),
    ("miriad", run_miriad_rag_search),
]

# image_path 必选的源 (旧纯函数实现里 image_path 为空会直接报错, 统一层先跳过)
_IMAGE_REQUIRED = {"medpix2", "radimagenet"}

# 打分参数: 文本分权重 1.0, 图像分作为辅分
_IMAGE_SCORE_WEIGHT = 0.3
_SCORE_TOP_N = 3  # 取 top-N 命中计算相似度均值


def _score_result(result: dict[str, Any]) -> float:
    """给一个旧工具的返回结果打分 (跨库可比)。

    score = mean(top-3 文本通道余弦相似度) + 0.3 × mean(top-3 图像通道余弦相似度)

    similarity 字段结构:
    - medpix2 / radimagenet: {"text": ..., "image": ...}
    - miriad:                {"question": ..., "answer": ...}
    除 "image" 外的通道均视为文本通道。
    """
    if not result or not result.get("success"):
        return 0.0
    hits = result.get("hits") or []
    if not hits:
        return 0.0

    text_sims: list[float] = []
    image_sims: list[float] = []
    for h in hits[:_SCORE_TOP_N]:
        sim = h.get("similarity") or {}
        for ch, v in sim.items():
            if not isinstance(v, (int, float)) or v is None:
                continue
            if ch == "image":
                image_sims.append(float(v))
            else:
                text_sims.append(float(v))
    if not text_sims:
        # 无文本相似度(仅图像通道命中): 以加权图像分兜底, 避免被记成 0 分
        if image_sims:
            return round(float(_IMAGE_SCORE_WEIGHT * statistics.mean(image_sims)), 6)
        return 0.0

    score = statistics.mean(text_sims)
    if image_sims:
        score += _IMAGE_SCORE_WEIGHT * statistics.mean(image_sims)
    return round(float(score), 6)


def _summarize_result(key: str, result: dict[str, Any]) -> dict[str, Any]:
    """压缩一份旧工具 result 为调试友好的摘要 (放 result["per_source"] 里)。"""
    return {
        "key": key,
        "success": bool(result.get("success")),
        "total_hits": result.get("total_hits", 0),
        "message": result.get("message", ""),
        "warning": bool(result.get("warnings")),
    }


def run_unified_rag_search(
    query: str,
    image_path: Optional[str] = None,
    top_k: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """依次调用全部三个 RAG 工具, 统一打分, 返回得分最高那家的
    (content_blocks, result)。"""
    query = (query or "").strip()
    image_path = (image_path or "").strip()
    if not query:
        error_blocks = [{"type": "text",
                         "text": "Unified RAG search failed: query is empty."}]
        error_result = {
            "success": False, "query": query, "winner": None,
            "scores": {}, "per_source": [], "total_hits": 0,
            "hits": [], "warnings": [],
            "message": "query is empty; pass non-empty search keywords.",
        }
        return error_blocks, error_result

    warnings: list[str] = []
    per_source: dict[str, dict[str, Any]] = {}  # key -> 旧工具原始 result
    per_source_blocks: dict[str, list[dict[str, Any]]] = {}  # key -> content_blocks
    scores: dict[str, float] = {}

    for key, fn in _SOURCES:
        try:
            if key in _IMAGE_REQUIRED:
                if not image_path:
                    # 无查询图: 旧工具 image_path 必选, 直接跳过, 不参与仲裁。
                    per_source[key] = {
                        "success": False, "total_hits": 0,
                        "message": "skipped: no query image provided",
                    }
                    scores[key] = 0.0
                    warnings.append(
                        f"No query image provided; image-based RAG '{key}' skipped."
                    )
                    continue
                if not os.path.isfile(image_path):
                    per_source[key] = {
                        "success": False, "total_hits": 0,
                        "message": f"skipped: query image not found '{image_path}'",
                    }
                    scores[key] = 0.0
                    warnings.append(
                        f"Query image not found '{image_path}'; RAG '{key}' skipped."
                    )
                    continue
                blocks, result = fn(query, image_path=image_path, top_k=top_k)
            else:
                blocks, result = fn(query=query, top_k=top_k)
        except Exception as e:  # 兜底: 单库异常不拖垮整体; 旧工具内部一般不抛
            logger.exception("Unified RAG: source '%s' raised", key)
            per_source[key] = {
                "success": False, "total_hits": 0,
                "message": f"raised {type(e).__name__}: {e}",
            }
            scores[key] = 0.0
            warnings.append(f"RAG '{key}' failed: {type(e).__name__}: {e}")
            continue

        per_source[key] = result
        per_source_blocks[key] = blocks
        scores[key] = _score_result(result)

    # 仲裁: 得分最高者为 winner (并列取先注册者)。
    winner_key = max(scores, key=scores.get) if scores else None
    winner_score = scores.get(winner_key, 0.0) if winner_key else 0.0
    winner_result = per_source.get(winner_key) if winner_key else None

    if winner_key is None or not winner_result or not winner_result.get("success"):
        # 所有库都不可用 / 无命中
        error_blocks = [{"type": "text",
                         "text": "Unified RAG search returned no usable results."}]
        error_result = {
            "success": False, "query": query, "image_path": image_path or None,
            "winner": None, "winner_score": 0.0, "scores": scores,
            "per_source": [
                _summarize_result(k, v) for k, v in per_source.items()
            ],
            "total_hits": 0, "hits": [], "warnings": warnings,
            "message": "All RAG sources unavailable or returned no hits.",
        }
        return error_blocks, error_result

    # 只下发 winner 的 content_blocks, 顶部加一行来源/得分说明。
    winner_blocks = per_source_blocks.get(winner_key, [])
    overview_lines = [
        f"## Unified RAG Results (winner: {winner_key}, score {winner_score:.3f})",
        f"- Query: {query}",
    ]
    if image_path:
        overview_lines.append(f"- Query image: {image_path}")
    content_blocks: list[dict[str, Any]] = [
        {"type": "text", "text": "\n".join(overview_lines) + "\n"},
    ]
    content_blocks.extend(winner_blocks)

    # 构造 result (hits 从 winner 的 result 透传)。
    winner_hits = (winner_result.get("hits") or [])[:top_k]
    result: dict[str, Any] = {
        "success": True,
        "query": query,
        "image_path": image_path or None,
        "winner": winner_key,
        "winner_score": winner_score,
        "scores": {k: scores[k] for k, _ in _SOURCES if k in scores},
        "per_source": [
            _summarize_result(k, per_source[k]) for k, _ in _SOURCES
        ],
        "total_hits": len(winner_hits),
        "hits": winner_hits,
        "warnings": warnings,
        "message": (
            f"Retrieved from unified RAG: winner '{winner_key}' "
            f"(score {winner_score:.3f}); "
            f"per-source scores { {k: scores[k] for k in scores} }."
        ),
    }
    return content_blocks, result


# ---------------------------------------------------------------------------
# LangChain tool
# ---------------------------------------------------------------------------

class UnifiedRagSearchInput(BaseModel):
    query: str = Field(
        description=(
            "Required. Search keywords describing the medical finding, anatomy, "
            "or question of interest, e.g. 'pulmonary embolism filling defect' "
            "or 'what causes chest pain'. Used for text-vector retrieval across "
            "all knowledge bases."
        ),
        min_length=1,
    )
    image_path: Optional[str] = Field(
        default=None,
        description=(
            "Optional. Absolute path to the query image file (PNG/JPG). "
            "Provide it whenever an image is available to enable image-vector "
            "retrieval against the imaging knowledge bases; omit it when the "
            "query concerns text-only medical knowledge (e.g. literature QA)."
        ),
    )
    top_k: Optional[int] = Field(
        default=5,
        description="Optional. Max number of fused results to return from the "
                    "winning source (default 5).",
    )


@tool(args_schema=UnifiedRagSearchInput)
def retrieve_unified_knowledge(
    query: str,
    image_path: Optional[str] = None,
    top_k: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    If the Task Type is "medical", calling this tool is FORBIDDEN. — use retrieve_rule instead. Violating calls are invalid and harmful.

    Medical Image Finding Retrieval: Queries specialized image interpretation databases
    (MedPix-2.0 clinical imaging descriptions, RadImageNet radiology VQA, and MIRIAD
    literature QA) to retrieve relevant visual findings and radiological observations.
    The system performs multi-source retrieval with quality scoring and returns the
    top-ranked reference findings.

    Inputs:
    - query: Search keywords for the finding/anatomy/question (required)
    - image_path: Optional path to the query image (PNG/JPG); provide it to
      enable image-based retrieval, omit it for text-only medical knowledge.
    - top_k: Optional max results (default 5)

    Outputs (content_blocks → visible to LLM):
    - Text: Overview (winning source + score) followed by the winning source's
      ranked hits; reference images included when the winner has them.

    Outputs (result dict → application metadata):
    - success / winner / winner_score / scores (per-source scores)
    - hits: winning source's hits (same shape as its underlying tool)
    - per_source: per-source status summary (success / total_hits / message)
    - warnings / message
    """
    # [CALL-LOG] 每次被调用都留痕: 哪个题目(sample_id+claim) + 什么参数 + 结果。
    # 当前题目由 run_multi_agent _run_single_sample 用 set_current_case_context() 注入。
    record: dict[str, Any] = {
        "event": "retrieve_unified_knowledge",
        "call_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    record.update(_current_case())
    record["args"] = {"query": query, "image_path": image_path, "top_k": top_k}
    try:
        content_blocks, result = run_unified_rag_search(
            query=query, image_path=image_path, top_k=top_k
        )
    except Exception:
        record["result"] = {"success": False}
        _log_rag_call(record)
        raise
    record["result"] = {
        "success": bool(result.get("success")),
        "winner": result.get("winner"),
        "winner_score": result.get("winner_score"),
        "total_hits": result.get("total_hits", 0),
        "per_source": result.get("per_source") or [],
        "warnings": bool(result.get("warnings")),
        "message": result.get("message", ""),
    }
    _log_rag_call(record)
    return content_blocks, result