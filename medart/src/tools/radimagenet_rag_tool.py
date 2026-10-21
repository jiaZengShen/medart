"""
RadImageNet-VQA RAG search tool for LangChain agent.

放射学 VQA(RadImageNet-VQA)多模态检索工具:
- 输入: 检索问题/关键词(必选)+ 图像路径(必选)+ question_id(可选)
- 流程: 关键词 → 文本向量检索(qa_vec); 图像 → 图像向量检索(image_vec);
  两路结果经 RRF(Reciprocal Rank Fusion)融合排序, 返回 QA 文本 + 图像。
- Milvus 单机服务器(默认 http://127.0.0.1:19530), 与 import_radimagenet_vqa.py
  导入的两个 collection 对应: radimagenet_qa(qa 文本) / radimagenet_img(图像),
  同一 QA 样本的两个 collection 主键 id 相同。

与 multimodal_rag_tool.py(medpix2)同架构, 并【复用】其基础设施:
- 编码器注册表(PubMedBERT + MedImageInsight)共享 multimodal 那份 —— 避免
  同一模型被加载两份占用显存。多工具一起用时不重复占 GPU。
- Milvus 客户端懒加载单例 + collection load 状态跟踪
- FIFO 公平门闸(多智能体并发调用时同一时刻只允许一个请求做编码/检索,
  防止并发把 GPU 显存/内存叠加打爆 OOM)

差异点(本数据集特有):
- image 字段存的是完整图片路径(非文件名), 直接用作 data URL 来源
- 检索集合: radimagenet_qa / radimagenet_img
- 过滤字段是 question_id

注册到 agent: 在 run_multi_agent.py 中
    from src.tools.radimagenet_rag_tool import retrieve_radimagenet_knowledge
    AVAILABLE_TOOLS = [..., retrieve_radimagenet_knowledge]
"""

from __future__ import annotations

import logging
import os
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from langchain.tools import tool
from pydantic import BaseModel, Field

from src.multi_agent.image_utils import build_image_data_url
from src.tools.multimodal_rag_tool import (
    _ensure_collection_loaded,
    _get_client,
    _RAG_GATE,
    get_image_encoder,
    get_text_encoder,
)
# [EVIDENCE-FILTER] 实验性证据过滤(可整体移除): 详见 rag_evidence_filter.py
from src.tools.rag_evidence_filter import filter_rag_hits_by_evidence

logger = logging.getLogger("radimagenet_rag_tool")
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )


# 与导入脚本保持一致的向量维度
IMAGE_VEC_DIM = 1024   # MedImageInsight (image_vec)
QA_VEC_DIM = 768       # PubMedBERT (qa_vec)


# ---------------------------------------------------------------------------
# RAG 调用观察者（供调试报告记录请求参数与返回结果）
# ---------------------------------------------------------------------------

_RAG_OBSERVERS: list[Callable[[dict[str, Any], dict[str, Any]], None]] = []


def register_rag_observer(observer: Callable[[dict[str, Any], dict[str, Any]], None]) -> None:
    _RAG_OBSERVERS.append(observer)


def _notify_rag_observers(request_info: dict[str, Any], result: dict[str, Any]) -> None:
    for obs in _RAG_OBSERVERS:
        try:
            obs(request_info, result)
        except Exception:
            logger.exception("RAG observer %r failed", obs)


def _rag_request_info(query: str, image_path: str, question_id: Optional[str],
                      dataset: Optional[str], top_k: int) -> dict[str, Any]:
    return {
        "query": query,
        "image_path": image_path,
        "question_id": question_id,
        "dataset": dataset,
        "top_k": top_k,
    }


# ---------------------------------------------------------------------------
# Dataset / Channel 注册表
# ---------------------------------------------------------------------------


@dataclass
class ChannelSpec:
    collection_name: str
    vec_field: str        # 向量字段名, 如 "image_vec" / "qa_vec"
    dim: int              # 向量维度, 如 1024 / 768


@dataclass
class DatasetSpec:
    """一个数据集 = 文本通道 + 图像通道 两个 collection。"""

    key: str
    description: str
    text_channel: ChannelSpec
    image_channel: ChannelSpec
    text_field: str = "qa"               # 文本描述字段(VQA 文本)
    image_field: str = "image"           # 图像字段(存完整路径)
    image_is_full_path: bool = True      # image 字段已是完整路径, 无需 roots 拼
    case_id_field: Optional[str] = "question_id"   # 过滤字段
    text_encoder_key: str = "pubmedbert"
    image_encoder_key: str = "medimageinsight"
    extra_output_fields: list[str] = field(default_factory=list)
    channel_top_k: int = 10
    rrf_k: int = 60


_DATASETS: dict[str, DatasetSpec] = {}
_DEFAULT_DATASET: Optional[str] = None


def register_dataset(spec: DatasetSpec, default: bool = False) -> None:
    _DATASETS[spec.key] = spec
    global _DEFAULT_DATASET
    if default or _DEFAULT_DATASET is None:
        _DEFAULT_DATASET = spec.key


def get_dataset(key: Optional[str] = None) -> DatasetSpec:
    k = key or _DEFAULT_DATASET
    if k is None or k not in _DATASETS:
        available = ", ".join(sorted(_DATASETS)) or "(none)"
        raise KeyError(f"Unknown dataset '{k}'. Available: {available}")
    return _DATASETS[k]


def list_datasets() -> list[dict[str, Any]]:
    return [
        {
            "key": s.key,
            "description": s.description,
            "text_collection": s.text_channel.collection_name,
            "image_collection": s.image_channel.collection_name,
        }
        for s in _DATASETS.values()
    ]


# ---------------------------------------------------------------------------
# 默认注册: radimagenet collection（编码器复用 multimodal 的注册表）
# ---------------------------------------------------------------------------

register_dataset(
    DatasetSpec(
        key="radimagenet",
        description="RadImageNet-VQA 放射学 CT/MRI 视觉问答数据集",
        text_channel=ChannelSpec(
            collection_name="radimagenet_qa",
            vec_field="qa_vec",
            dim=QA_VEC_DIM,
        ),
        image_channel=ChannelSpec(
            collection_name="radimagenet_img",
            vec_field="image_vec",
            dim=IMAGE_VEC_DIM,
        ),
        text_field="qa",
        image_field="image",
        image_is_full_path=True,
        case_id_field="question_id",
        text_encoder_key="pubmedbert",
        image_encoder_key="medimageinsight",
        extra_output_fields=[
            "content_type", "correct_text", "is_abnormal",
            "location", "modality", "pathology", "question_id",
        ],
        channel_top_k=10,
    ),
    default=True,
)


# ---------------------------------------------------------------------------
# 检索核心
# ---------------------------------------------------------------------------


def _build_filter(spec: DatasetSpec, question_id: Optional[str],
                  warnings: list[str]) -> str:
    qid = (question_id or "").strip().strip('"')
    if not qid:
        return ""
    if not spec.case_id_field:
        warnings.append(
            f"Dataset '{spec.key}' has no case_id field; "
            f"question_id filter '{qid}' ignored."
        )
        return ""
    return f'{spec.case_id_field} == "{qid}"'


def _resolve_image_path(spec: DatasetSpec, image_field_value: str) -> Optional[str]:
    """解析命中图片的磁盘路径。

    radimagenet 的 image 字段存的是完整路径(image_is_full_path=True),
    直接校验存在即可。
    """
    if image_field_value and spec.image_is_full_path:
        return image_field_value if os.path.isfile(image_field_value) else None
    return None


def _search_channel(spec: DatasetSpec, channel: ChannelSpec, vec: np.ndarray,
                    flt: str, top_k: int) -> list[dict[str, Any]]:
    """在单个通道 collection 上做向量检索并归一化命中结果。

    Milvus 客户端来自 multimodal 的共享单例; 若连接被服务端掐断
    (如 gRPC GOAWAY), _get_client 内部会重建重试。"""
    output_fields = list(dict.fromkeys(
        [spec.text_field, spec.image_field,
         *(spec.extra_output_fields or [])]
    ))
    client = _get_client()
    _ensure_collection_loaded(client, channel.collection_name)
    res = client.search(
        collection_name=channel.collection_name,
        data=[vec.tolist()],
        anns_field=channel.vec_field,
        search_params={"metric_type": "COSINE", "params": {}},
        limit=top_k,
        filter=flt,
        output_fields=output_fields,
    )

    hits = []
    for hit in (res[0] if res else []):
        entity = hit.get("entity", {}) if isinstance(hit, dict) else hit.entity
        distance = hit.get("distance") if isinstance(hit, dict) else hit.distance
        pk = hit.get("id") if isinstance(hit, dict) else hit.id
        img = entity.get(spec.image_field, "")
        hits.append({
            "key": img or str(pk),  # 去重键: 优先图片路径
            "image": img,
            "text": entity.get(spec.text_field, "") or "",
            "similarity": float(distance) if distance is not None else None,
            "fields": {k: v for k, v in entity.items()},
        })
    return hits


def _rrf_merge(text_hits: list[dict], image_hits: list[dict],
               rrf_k: int = 60) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion 融合两路召回, 同 key(图片路径)合并。"""
    merged: dict[str, dict[str, Any]] = {}
    for channel, hits in (("text", text_hits), ("image", image_hits)):
        for rank, h in enumerate(hits):
            m = merged.setdefault(h["key"], {
                "key": h["key"], "image": h["image"], "text": h["text"],
                "fields": h["fields"], "rrf_score": 0.0,
                "similarity": {}, "sources": [],
            })
            m["rrf_score"] += 1.0 / (rrf_k + rank + 1)
            m["similarity"][channel] = h["similarity"]
            m["sources"].append(channel)
            if not m["text"] and h["text"]:
                m["text"] = h["text"]
    return sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)


def preload_rag_models(dataset: Optional[str] = None) -> None:
    """预加载本工具用到的编码器(复用 multimodal 共享注册表, 幂等)。

    由于与 multimodal/miriad 共享同一份模型, 重复调用不会额外占用显存。
    """
    spec = get_dataset(dataset)
    get_text_encoder(spec.text_encoder_key)
    get_image_encoder(spec.image_encoder_key)
    logger.info(
        "RadImageNet RAG encoders preloaded (shared registry): text=%s, image=%s",
        spec.text_encoder_key, spec.image_encoder_key,
    )


# ---------------------------------------------------------------------------
# 纯函数入口（流水线可直接调用, 不依赖 langchain）
# ---------------------------------------------------------------------------


def run_radimagenet_rag_search(
    query: str,
    image_path: str,
    question_id: Optional[str] = None,
    dataset: Optional[str] = None,
    top_k: int = 5,
    max_images: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """双路检索: 关键词→文本(qa_vec), 图像→图像(image_vec), RRF 融合。
    返回 (content_blocks, result)。"""

    def _error_return(msg: str, tb: Optional[str] = None):
        text = f"RadImageNet RAG search failed: {msg}"
        if tb:
            text += f"\n\nFull traceback:\n{tb}"
        blocks = [{"type": "text", "text": text}]
        result = {
            "success": False, "dataset": dataset, "query": query,
            "question_id": question_id, "total_hits": 0, "n_text_hits": 0,
            "n_image_hits": 0, "hits": [], "warnings": [], "message": msg,
            "traceback": tb or "",
        }
        return blocks, result

    # 1. 输入校验
    query = (query or "").strip()
    image_path = (image_path or "").strip()
    if not query:
        return _error_return("query is empty; pass non-empty search keywords.")
    if not image_path:
        return _error_return("image_path is empty; pass the query image path.")
    if not os.path.isfile(image_path):
        return _error_return(f"Query image not found: '{image_path}'")

    warnings: list[str] = []
    try:
        spec = get_dataset(dataset)
    except KeyError as e:
        return _error_return(str(e))

    # 2. 双路检索
    stage = "preparing filter"
    try:
        flt = _build_filter(spec, question_id, warnings)

        with _RAG_GATE:
            stage = f"text encoding ({spec.text_encoder_key})"
            text_enc = get_text_encoder(spec.text_encoder_key)
            q_vec = text_enc.encode_texts([query])[0]

            stage = (f"text vector search "
                     f"({spec.text_channel.collection_name}.{spec.text_channel.vec_field})")
            text_hits = _search_channel(spec, spec.text_channel, q_vec, flt,
                                        spec.channel_top_k)

            stage = f"image encoding ({spec.image_encoder_key})"
            img_enc = get_image_encoder(spec.image_encoder_key)
            i_vec = img_enc.encode_images([image_path])[0]

            stage = (f"image vector search "
                     f"({spec.image_channel.collection_name}.{spec.image_channel.vec_field})")
            image_hits = _search_channel(spec, spec.image_channel, i_vec, flt,
                                         spec.channel_top_k)
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("RadImageNet RAG retrieval failed at stage [%s]", stage)
        error_blocks, error_result = _error_return(
            f"retrieval error at [{stage}]: {type(e).__name__}: {e}", tb=tb)
        _notify_rag_observers(
            _rag_request_info(query, image_path, question_id, dataset, top_k),
            error_result,
        )
        return error_blocks, error_result

    # 3. 融合 + 解析图片路径
    fused = _rrf_merge(text_hits, image_hits, spec.rrf_k)[:top_k]
    # [EVIDENCE-FILTER] 实验性: 仅保留支持查询的证据, 非证据命中不下发(可移除)
#   注意: 图像驱动 (以查询图检索库内图) 的 RAG 整批放行; 只对纯文本召回过滤
    fused = filter_rag_hits_by_evidence(
        query, fused, text_key="text",
        collection=f"{spec.text_channel.collection_name} + "
                   f"{spec.image_channel.collection_name}",
        image_driven=True)
    for h in fused:
        h["image_path"] = _resolve_image_path(spec, h["image"])
        if h["image"] and not h["image_path"]:
            warnings.append(f"Image file not found for '{h['image']}'")

    # 4. content_blocks: 概览 + 逐条"文本块 紧贴 图片块"交错布局
    overview = [
        f"## RadImageNet-VQA RAG Results (dataset: {spec.key})\n",
        f"- Query keywords: {query}",
        f"- Query image: {image_path}",
        f"- question_id filter: {question_id or '(none)'}",
        f"- Text-channel hits: {len(text_hits)} | "
        f"Image-channel hits: {len(image_hits)} | "
        f"Fused results: {len(fused)}\n",
    ]
    content_blocks: list[dict[str, Any]] = [
        {"type": "text", "text": "\n".join(overview)},
    ]
    n_images = 0
    for i, h in enumerate(fused, 1):
        sim = ", ".join(
            f"{ch}={h['similarity'].get(ch):.3f}"
            for ch in h["sources"] if h["similarity"].get(ch) is not None
        )
        extra = {k: h["fields"].get(k) for k in (spec.extra_output_fields or [])
                 if h["fields"].get(k) not in (None, "")}
        hit_lines = [
            f"### Hit #{i}  [{'+'.join(h['sources'])} | sim: {sim} | "
            f"rrf: {h['rrf_score']:.4f}]",
            f"- image: {h['image']}",
        ]
        if extra:
            hit_lines.append(f"- meta: {extra}")
        hit_lines.append(f"- qa: {h['text'] or '(no qa text)'}")
        content_blocks.append({"type": "text", "text": "\n".join(hit_lines)})

        if h.get("image_path") and n_images < max_images:
            try:
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": build_image_data_url(h["image_path"])},
                })
                n_images += 1
            except Exception as e:
                warnings.append(f"Failed to encode image '{h['image_path']}': {e}")

    if warnings:
        content_blocks.append({
            "type": "text",
            "text": "**Warnings**:\n" + "\n".join(f"- {w}" for w in warnings),
        })

    # 5. result 元数据
    result: dict[str, Any] = {
        "success": True,
        "dataset": spec.key,
        "collections": [
            spec.text_channel.collection_name,
            spec.image_channel.collection_name,
        ],
        "query": query,
        "question_id": question_id,
        "total_hits": len(fused),
        "n_text_hits": len(text_hits),
        "n_image_hits": len(image_hits),
        "hits": [
            {
                "rank": i,
                "dataset": spec.key,
                "image": h["image"],
                "image_path": h.get("image_path"),
                "text": h["text"],
                "sources": h["sources"],
                "similarity": h["similarity"],
                "rrf_score": round(h["rrf_score"], 6),
                "fields": h["fields"],
            }
            for i, h in enumerate(fused, 1)
        ],
        "warnings": warnings,
        "message": (
            f"Retrieved {len(fused)} fused result(s) from dataset '{spec.key}' "
            f"({n_images} image(s) attached)."
        ),
    }
    _notify_rag_observers(
        _rag_request_info(query, image_path, question_id, dataset, top_k),
        result,
    )
    return content_blocks, result


# ---------------------------------------------------------------------------
# LangChain tool
# ---------------------------------------------------------------------------


class RadImageNetRagSearchInput(BaseModel):
    query: str = Field(
        description=(
            "Required. Search keywords describing the radiologic finding or "
            "anatomy, e.g. 'pulmonary embolism' or 'liver lesion'. Used for "
            "text-vector retrieval against the QA text."
        ),
        min_length=1,
    )
    image_path: str = Field(
        description=(
            "Required. Absolute path to the query image file (PNG/JPG). "
            "Used for content-based image-vector retrieval."
        ),
        min_length=1,
    )
    question_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional. Question template id (e.g. 'pathology_mc') to restrict "
            "retrieval to samples of the same question type / template."
        ),
    )
    dataset: Optional[str] = Field(
        default=None,
        description=(
            "Optional. Dataset key to search (default: the default "
            "collection). Available datasets can be listed via list_datasets()."
        ),
    )
    top_k: Optional[int] = Field(
        default=5,
        description="Optional. Max number of fused results to return (default 5).",
    )


@tool(args_schema=RadImageNetRagSearchInput)
def retrieve_radimagenet_knowledge(
    query: str,
    image_path: str,
    question_id: Optional[str] = None,
    dataset: Optional[str] = None,
    top_k: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """从 RadImageNet-VQA 放射学知识库中检索参考资料（QA 文本 + 图像）。
    RadImageNet-VQA is a large-scale radiologic VQA dataset built on CT/MRI:
    750K expert-curated images paired with 7.5M QA samples across 8 anatomical
    regions and 97 pathologies, covering abnormality detection, anatomy
    recognition, and fine-grained pathology identification, in open-ended,
    closed-ended, and multiple-choice formats.

    Search the RadImageNet-VQA radiologic knowledge base with BOTH text and
    image channels: the keywords are embedded and matched against QA text
    vectors (radimagenet_qa), while the query image is embedded and matched
    against stored images (radimagenet_img); the two result lists are fused
    (RRF) and the top hits are returned as QA text plus reference images.

    Inputs:
    - query: Search keywords for the finding/anatomy (required)
    - image_path: Path to the query image file (required)
    - question_id: Optional question template id to filter within one type
    - dataset: Optional dataset key (defaults to 'radimagenet')
    - top_k: Optional max fused results (default 5)

    Outputs (content_blocks → visible to LLM):
    - Text: Ranked hits with QA text, source channel(s), similarities
    - Image(s): Reference images of the hits as data URLs

    Outputs (result dict → application metadata):
    - success / total_hits / n_text_hits / n_image_hits
    - hits: [{rank, dataset, image, image_path, text, sources,
      similarity, rrf_score, fields}]
    - warnings / message
    """
    return run_radimagenet_rag_search(
        query=query,
        image_path=image_path,
        question_id=question_id,
        dataset=dataset,
        top_k=top_k or 5,
    )