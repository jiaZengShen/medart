"""
Multimodal RAG search tool for LangChain agent.

多模态医学影像 RAG 检索工具：
- 输入：检索关键词（必选）+ 图像路径（必选）+ case_id（可选）
- 流程：关键词 → 文本向量检索（caption_vec）；图像 → 图像向量检索（image_vec）；
  两路结果经 RRF（Reciprocal Rank Fusion）融合排序，返回文本描述 + 图像。
- Milvus 单机服务器（http://127.0.0.1:19530），非 Lite。

可扩展性设计（后期接入新数据集无需改动检索逻辑）：
- Dataset 注册表：一个数据集 = 文本/图像两个 collection（每个只含一个向量字段,
  规避 milvus-lite 对混合维度多向量字段的索引加载 bug）。每份 DatasetSpec
  配置两个 ChannelSpec（collection 名、向量字段、维度）+ 文本字段、case_id
  字段、图片目录、所用编码器等；通过 register_dataset() 注册, 工具按
  dataset key 路由, 两路召回以 image 名跨库融合。
- Encoder 注册表：文本/图像编码器分别按 key 注册工厂函数，懒加载 + 单例 +
  线程安全；新模型实现相同接口后 register_text_encoder /
  register_image_encoder 一行接入。
- 路径多候选：模型目录 / 数据集目录支持环境变量覆盖 + 多盘符候选，
  换机器无需改代码。

遵循项目 tool 约定
- @tool + Pydantic args_schema
- 返回 (content_blocks, result) 元组：content_blocks 含文本块 + 图片 data URL，
  多模态 LLM 可直接读到/看到；result 为应用层元数据。
- 暴露纯函数 run_multimodal_rag_search()，不依赖 langchain，可被流水线直接调用。

注册到 agent：在 run_multi_agent.py 中
    from src.tools.multimodal_rag_tool import retrieve_medical_knowledge
    AVAILABLE_TOOLS = [..., retrieve_medical_knowledge]

注意：两个编码器首次调用时加载并常驻显存（PubMedBERT ~0.5GB，
MedImageInsight 数 GB），之后的每次工具调用直接复用。

并发保护：工具入口带 FIFO 公平门闸 —— 多智能体多线程并发调用时,
同一时刻只允许一个请求做向量编码/检索, 其余按到达顺序排队, 防止
并发把 GPU 显存/内存叠加打爆 (OOM)。
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import tempfile
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from langchain.tools import tool
from pydantic import BaseModel, Field

from src.multi_agent.image_utils import build_image_data_url
# [EVIDENCE-FILTER] 实验性证据过滤(可整体移除): 详见 rag_evidence_filter.py
from src.tools.rag_evidence_filter import filter_rag_hits_by_evidence

logger = logging.getLogger("multimodal_rag_tool")
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# FIFO 公平互斥锁（多智能体并发保护, 防 OOM）
# ---------------------------------------------------------------------------


class _FifoSemaphore:
    """严格先到先服务的互斥信号量: 同一时刻只允许一个 RAG 请求通过。

    多智能体框架是多线程并发调用工具, 若多个请求同时做向量编码/检索,
    MedImageInsight 等常驻显存的模型加上各请求自己的中间张量叠加,
    容易把 GPU 显存打爆 (OOM)。这里用「票号」方案实现严格 FIFO:
      - 每个请求到达时领一个单调递增的票号;
      - 当前票号落到 [serving, serving+capacity) 区间内才放行;
      - 结束后 serving 前进一步, 恰好把下一个排队的请求放进来。
    Python 内置的 threading.Lock / Semaphore 只保证「大致公平」, 高并发
    竞争下可能出现后来的请求先抢到锁, 不满足 FIFO 需求, 故手动实现。

    capacity 默认 1(互斥)。用法:
        with _RAG_GATE:
            ... 编码 + 检索 ...
    """

    def __init__(self, capacity: int = 1):
        self._capacity = capacity
        self._cond = threading.Condition()
        self._next_ticket = 0  # 已发放的最大票号
        self._serving = 0      # 当前放行的区间起点票号

    def __enter__(self):
        with self._cond:
            ticket = self._next_ticket
            self._next_ticket += 1
            if ticket - self._serving >= self._capacity:
                logger.info(
                    "RAG FIFO gate: request #%d queued behind serving #%d ...",
                    ticket, self._serving,
                )
            while ticket - self._serving >= self._capacity:
                self._cond.wait()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with self._cond:
            self._serving += 1
            self._cond.notify_all()
        return False


# 进程级 RAG 门闸: 所有线程共享, 保证同一时刻只有一个请求在使用 RAG。
_RAG_GATE = _FifoSemaphore(capacity=1)


# ---------------------------------------------------------------------------
# RAG 调用观察者（供调试报告记录请求参数与返回结果）
# ---------------------------------------------------------------------------

_RAG_OBSERVERS: list[Callable[[dict[str, Any], dict[str, Any]], None]] = []


def register_rag_observer(observer: Callable[[dict[str, Any], dict[str, Any]], None]) -> None:
    """注册一个 RAG 调用观察者: 每次 RAG 调用结束后收到
    (request_info, result) 两个字典。可多次注册; 观察者内部异常会被
    吞掉并记日志, 不影响检索主流程。"""
    _RAG_OBSERVERS.append(observer)


def _notify_rag_observers(request_info: dict[str, Any], result: dict[str, Any]) -> None:
    for obs in _RAG_OBSERVERS:
        try:
            obs(request_info, result)
        except Exception:
            logger.exception("RAG observer %r failed", obs)


def _rag_request_info(query: str, image_path: str, case_id: Optional[str],
                      dataset: Optional[str], top_k: int) -> dict[str, Any]:
    """汇总一次 RAG 请求参数（LLM 传给工具的参数, 供调试报告展示）。"""
    return {
        "query": query,
        "image_path": image_path,
        "case_id": case_id,
        "dataset": dataset,
        "top_k": top_k,
    }


# ---------------------------------------------------------------------------
# 路径解析（环境变量覆盖 + 多候选盘符）
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _first_existing(candidates: list[Optional[str]]) -> Optional[str]:
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return None


def resolve_model_root() -> Optional[str]:
    return _first_existing([
        os.environ.get("RAG_MODEL_ROOT"),
        r"/home/jzs/od/model",
        r"I:\icdInformations\model",
    ])


def resolve_milvus_uri() -> str:
    """Milvus 单机服务器地址。默认 http://127.0.0.1:19530, 可用 RAG_MILVUS_URI
    覆盖(指向运行了 Milvus 单机版的机器)。"""
    env = os.environ.get("RAG_MILVUS_URI")
    if env:
        return env
    return "http://127.0.0.1:19530"


# ---------------------------------------------------------------------------
# Dataset / Channel 注册表
# ---------------------------------------------------------------------------

# 与导入脚本保持一致: medpix2 的向量维度
IMAGE_VEC_DIM = 1024   # MedImageInsight (DIM_PROJECTION)
CAPTION_VEC_DIM = 768  # PubMedBERT base


@dataclass
class ChannelSpec:
    """一个向量检索通道 = 一个单向量字段的 collection。

    milvus-lite 对"一个 collection 含多个不同维度向量字段"的索引加载有 bug,
    因此一个数据集拆成两个 collection, 每个只含一个向量字段、单一维度。
    """

    collection_name: str
    vec_field: str        # 向量字段名, 如 "image_vec" / "caption_vec"
    dim: int              # 向量维度, 如 1024 / 768


@dataclass
class DatasetSpec:
    """一个数据集 = 文本通道 + 图像通道 两个 collection。新增数据集 = 注册一份本配置。"""

    key: str                              # 数据集标识, 如 "medpix2"
    description: str                      # 给 LLM/开发者看的说明
    text_channel: ChannelSpec             # 文本向量通道 collection
    image_channel: ChannelSpec            # 图像向量通道 collection
    text_field: str = "caption"           # 文本描述字段(返回给 LLM)
    image_name_field: str = "image"       # 跨库关联键: 图片名字段(无扩展名)
    image_ext: str = ".png"               # 图片扩展名
    image_roots: list[str] = field(default_factory=list)  # 图片目录候选
    case_id_field: Optional[str] = None   # 支持 case_id 过滤的字段, 无则 None
    case_id_normalizer: Optional[Callable[[str], str]] = None  # case_id 归一化(可选)
    text_encoder_key: str = "pubmedbert"  # 使用哪个已注册文本编码器
    image_encoder_key: str = "medimageinsight"
    extra_output_fields: list[str] = field(default_factory=list)  # 附带返回字段
    channel_top_k: int = 10               # 每路检索召回数
    rrf_k: int = 60                       # RRF 融合参数


_DATASETS: dict[str, DatasetSpec] = {}
_DEFAULT_DATASET: Optional[str] = None


def register_dataset(spec: DatasetSpec, default: bool = False) -> None:
    """注册一个数据集(双 collection)。default=True 设为默认检索目标。"""
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
# Encoder 注册表（懒加载单例, 线程安全）
# ---------------------------------------------------------------------------

_TEXT_ENCODER_FACTORIES: dict[str, Callable[[], "BaseTextEncoder"]] = {}
_IMAGE_ENCODER_FACTORIES: dict[str, Callable[[], "BaseImageEncoder"]] = {}
_LOADED_ENCODERS: dict[str, Any] = {}
_ENCODER_LOCK = threading.Lock()


class BaseTextEncoder:
    """文本编码器接口: encode_texts(texts) -> (N, dim) float32。"""

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class BaseImageEncoder:
    """图像编码器接口: encode_images(paths) -> (N, dim) float32。"""

    def encode_images(self, paths: list[str]) -> np.ndarray:
        raise NotImplementedError


def register_text_encoder(key: str, factory: Callable[[], BaseTextEncoder]) -> None:
    _TEXT_ENCODER_FACTORIES[key] = factory


def register_image_encoder(key: str, factory: Callable[[], BaseImageEncoder]) -> None:
    _IMAGE_ENCODER_FACTORIES[key] = factory


def get_text_encoder(key: str) -> BaseTextEncoder:
    return _get_encoder(f"text:{key}", _TEXT_ENCODER_FACTORIES, key)


def get_image_encoder(key: str) -> BaseImageEncoder:
    return _get_encoder(f"image:{key}", _IMAGE_ENCODER_FACTORIES, key)


def _get_encoder(cache_key: str, factories: dict, key: str):
    with _ENCODER_LOCK:
        if cache_key in _LOADED_ENCODERS:
            return _LOADED_ENCODERS[cache_key]
        if key not in factories:
            raise KeyError(
                f"No encoder registered under '{key}'. "
                f"Available: {sorted(factories)}"
            )
        print(f"[RAG] loading encoder '{key}' ...")
        enc = factories[key]()
        _LOADED_ENCODERS[cache_key] = enc
        return enc


# ---------------------------------------------------------------------------
# 内置编码器实现: PubMedBERT(文本) / MedImageInsight(图像)
# ---------------------------------------------------------------------------


class PubMedBertTextEncoder(BaseTextEncoder):
    """PubMedBERT + mean pooling（与入库时的 caption_vec 一致）。"""

    def __init__(self, model_dir: str, max_length: int = 256):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        # 文本编码器固定放 CPU：GTX 1050 只有 4GB 显存，
        # 把显存全部留给 2.3GB 的 MedImageInsight 图像模型，避免两者抢显存。
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).to(self.device).eval()
        self.max_length = max_length
        print(f"[RAG] PubMedBERT loaded on {self.device}")

    @property
    def _no_grad(self):
        return self._torch.no_grad()

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        texts = [t if isinstance(t, str) and t.strip() else " " for t in texts]
        with self._no_grad:
            inputs = self.tokenizer(
                texts, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb = (outputs.last_hidden_state * mask).sum(dim=1)
            emb = emb / mask.sum(dim=1).clamp(min=1e-9)
        return emb.cpu().numpy().astype(np.float32)


def _make_pubmedbert() -> PubMedBertTextEncoder:
    root = resolve_model_root()
    if not root:
        raise FileNotFoundError(
            "Model root not found. Set RAG_MODEL_ROOT or place models under "
            r"C:\Users\jzs\icd\model or I:\icdInformations\model"
        )
    model_dir = os.path.join(root, "pubmedbert-base-embeddings")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"PubMedBERT dir not found: {model_dir}")
    return PubMedBertTextEncoder(model_dir)


def _patch_clip_tokenizer_compat() -> None:
    """兼容 transformers>=4.30 加载旧版 CLIP tokenizer 配置时的 AddedToken
    'special' 参数冲突: 拦截 CLIPTokenizer(Fast).from_pretrained, 将含
    "special" 键的目录重定向到去掉该键的本地副本（原子写入 + 损坏自愈）。"""
    import shutil
    from transformers import CLIPTokenizer, CLIPTokenizerFast

    if getattr(_patch_clip_tokenizer_compat, "_done", False):
        return
    _patch_clip_tokenizer_compat._done = True

    patched_root = os.path.join(tempfile.gettempdir(), "_clip_tokenizer_patched")
    cache: dict[str, str] = {}

    def _has_special_key(stm: dict) -> bool:
        def _check(v):
            return isinstance(v, dict) and "special" in v
        return any(
            (any(_check(i) for i in v) if isinstance(v, list) else _check(v))
            for v in stm.values()
        )

    def _sanitize(src_dir: str) -> str:
        if src_dir in cache:
            return cache[src_dir]
        stm_path = os.path.join(src_dir, "special_tokens_map.json")
        if not os.path.exists(stm_path):
            cache[src_dir] = src_dir
            return src_dir
        with open(stm_path, encoding="utf-8") as f:
            stm = json.load(f)
        if not _has_special_key(stm):
            cache[src_dir] = src_dir
            return src_dir

        dst = os.path.join(patched_root, os.path.basename(os.path.normpath(src_dir)))
        marker = os.path.join(dst, ".sanitized")
        if os.path.exists(marker):  # 校验缓存, 损坏则重建
            try:
                with open(os.path.join(dst, "special_tokens_map.json"), encoding="utf-8") as f:
                    json.load(f)
            except Exception:
                shutil.rmtree(dst, ignore_errors=True)
        if not os.path.exists(marker):
            os.makedirs(patched_root, exist_ok=True)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src_dir, dst)
            for v in stm.values():
                targets = v if isinstance(v, list) else [v]
                for item in targets:
                    if isinstance(item, dict):
                        item.pop("special", None)
            stm_dst = os.path.join(dst, "special_tokens_map.json")
            tmp = stm_dst + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(stm, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, stm_dst)
            with open(stm_dst, encoding="utf-8") as f:
                json.load(f)
            with open(marker, "w") as f:
                f.write("ok")
            print(f"[RAG] sanitized CLIP tokenizer -> {dst}")
        cache[src_dir] = dst
        return dst

    for cls in (CLIPTokenizer, CLIPTokenizerFast):
        orig = cls.from_pretrained.__func__

        def _make_wrapper(orig_fn):
            @classmethod
            def _from_pretrained(kls, pretrained_model_name_or_path, *args, **kwargs):
                if (isinstance(pretrained_model_name_or_path, str)
                        and os.path.isdir(pretrained_model_name_or_path)):
                    pretrained_model_name_or_path = _sanitize(pretrained_model_name_or_path)
                return orig_fn(kls, pretrained_model_name_or_path, *args, **kwargs)
            return _from_pretrained

        cls.from_pretrained = _make_wrapper(orig)


class MedImageInsightImageEncoder(BaseImageEncoder):
    """MedImageInsight 图像编码器（与入库时的 image_vec 一致）。

    编码前统一 convert("RGB"), 规避 RGBA/调色板 PNG 在模型内部
    Normalize 时 4 通道 vs 3 通道报错。"""

    def __init__(self, model_root_dir: str):
        import sys as _sys
        import torch
        from PIL import Image as PILImage

        _patch_clip_tokenizer_compat()
        if model_root_dir not in _sys.path:
            _sys.path.insert(0, model_root_dir)
        from medimageinsightmodel import MedImageInsight

        self._torch = torch
        self._PILImage = PILImage
        old_cwd = os.getcwd()
        os.chdir(model_root_dir)  # wrapper 内部用相对路径拼 config/权重
        try:
            self.classifier = MedImageInsight(
                model_dir="2024.09.27",
                vision_model_name="medimageinsigt-v1.0.0.pt",
                language_model_name="language_model.pth",
            )
            self.classifier.load_model()
            # 图像编码器强制固定到 GPU（文本编码器已移到 CPU）。
            # 正常情况下 load_model() 里已按 cuda 加载，这里仅作兜底。
            if torch.cuda.is_available() and self.classifier.device.type != "cuda":
                self.classifier.device = torch.device("cuda")
                self.classifier.model = self.classifier.model.to("cuda")
        finally:
            os.chdir(old_cwd)
        print(f"[RAG] MedImageInsight loaded on "
              f"{'cuda' if torch.cuda.is_available() else 'cpu'}")

    def encode_images(self, paths: list[str]) -> np.ndarray:
        b64_list = []
        for p in paths:
            with self._PILImage.open(p) as im:
                im = im.convert("RGB")
                buf = io.BytesIO()
                im.save(buf, format="PNG")
            b64_list.append(base64.encodebytes(buf.getvalue()).decode("utf-8"))
        out = self.classifier.encode(images=b64_list)
        return out["image_embeddings"].astype(np.float32)


def _make_medimageinsight() -> MedImageInsightImageEncoder:
    root = resolve_model_root()
    if not root:
        raise FileNotFoundError(
            "Model root not found. Set RAG_MODEL_ROOT or place models under "
            r"C:\Users\jzs\icd\model or I:\icdInformations\model"
        )
    model_dir = os.path.join(root, "MedImageInsights")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"MedImageInsights dir not found: {model_dir}")
    return MedImageInsightImageEncoder(model_dir)


# ---------------------------------------------------------------------------
# 默认注册: 编码器 + medpix2 collection
# ---------------------------------------------------------------------------


def _normalize_medpix_case_id(case_id: str) -> str:
    """MedPix 的 u_id 形如 'MPX2594', 但调用方可能传图像级 ID
    'MPX2594_synpic20734' —— 提取 MPX 前缀归一化到病例级。"""
    m = re.match(r"MPX\d+", case_id)
    return m.group(0) if m else case_id


register_text_encoder("pubmedbert", _make_pubmedbert)
register_image_encoder("medimageinsight", _make_medimageinsight)

register_dataset(
    DatasetSpec(
        key="medpix2",
        description="MedPix-2.0 放射学影像-描述数据集",
        text_channel=ChannelSpec(
            collection_name="med_pix2_text",
            vec_field="caption_vec",
            dim=CAPTION_VEC_DIM,
        ),
        image_channel=ChannelSpec(
            collection_name="med_pix2_image",
            vec_field="image_vec",
            dim=IMAGE_VEC_DIM,
        ),
        text_field="caption",
        image_name_field="image",
        image_ext=".png",
        image_roots=[
            os.environ.get("MEDPIX2_IMAGES_DIR"),
            r"C:\Users\jzs\icd\1数据集\MedPix-2.0\MedPix-2-0\images",
            r"I:\icdInformations\1数据集\MedPix-2.0\MedPix-2-0\images",
        ],
        case_id_field="u_id",
        case_id_normalizer=_normalize_medpix_case_id,
        text_encoder_key="pubmedbert",
        image_encoder_key="medimageinsight",
        extra_output_fields=["u_id", "type", "location", "location_category"],
        channel_top_k=10,
    ),
    default=True,
)


# ---------------------------------------------------------------------------
# Milvus 客户端（懒加载单例）
# ---------------------------------------------------------------------------

_MILVUS_CLIENT = None
_MILVUS_LOCK = threading.Lock()
_LOADED_COLLECTIONS: set[str] = set()


def _get_client():
    global _MILVUS_CLIENT
    with _MILVUS_LOCK:
        if _MILVUS_CLIENT is None:
            from pymilvus import MilvusClient
            uri = resolve_milvus_uri()
            _MILVUS_CLIENT = MilvusClient(uri)
            print(f"[RAG] Milvus connected: {uri}")
        return _MILVUS_CLIENT


def _ensure_collection_loaded(client, collection_name: str) -> None:
    """Milvus Lite 的 collection 每次进程启动默认处于 released 状态,
    search/query 前必须先 load。已加载过的跳过。"""
    if collection_name in _LOADED_COLLECTIONS:
        return
    client.load_collection(collection_name)
    _LOADED_COLLECTIONS.add(collection_name)


def _reset_client() -> bool:
    """丢弃失效的 Milvus 客户端并清空已加载标记(下次 _get_client 会重建重连)。"""
    global _MILVUS_CLIENT
    with _MILVUS_LOCK:
        _MILVUS_CLIENT = None
    _LOADED_COLLECTIONS.clear()
    return True


# ---------------------------------------------------------------------------
# 检索核心
# ---------------------------------------------------------------------------


def _build_filter(spec: DatasetSpec, case_id: Optional[str],
                  warnings: list[str]) -> str:
    cid = (case_id or "").strip().strip('"')
    if not cid:
        return ""
    if not spec.case_id_field:
        warnings.append(
            f"Dataset '{spec.key}' has no case_id field; "
            f"case_id filter '{cid}' ignored."
        )
        return ""
    if spec.case_id_normalizer:
        cid = spec.case_id_normalizer(cid)
    return f'{spec.case_id_field} == "{cid}"'


def _resolve_image_path(spec: DatasetSpec, image_name: str) -> Optional[str]:
    for root in spec.image_roots:
        if not root:
            continue
        p = os.path.join(root, image_name + spec.image_ext)
        if os.path.isfile(p):
            return p
    return None


def _search_channel(spec: DatasetSpec, channel: ChannelSpec, vec: np.ndarray,
                    flt: str, top_k: int) -> list[dict[str, Any]]:
    """在单个通道 collection 上做向量检索并归一化命中结果。

    若连接被服务端掐断(如 gRPC too_many_pings GOAWAY), 自动重建客户端
    重试一次, 避免搜索永久卡死在失效连接上。"""
    output_fields = list(dict.fromkeys(
        [spec.text_field, spec.image_name_field,
         *(spec.extra_output_fields or [])]
    ))
    last_exc: Optional[Exception] = None
    for attempt in range(2):
        client = _get_client()
        _ensure_collection_loaded(client, channel.collection_name)
        try:
            res = client.search(
                collection_name=channel.collection_name,
                data=[vec.tolist()],
                anns_field=channel.vec_field,
                search_params={"metric_type": "COSINE", "params": {}},
                limit=top_k,
                filter=flt,
                output_fields=output_fields,
            )
            break
        except Exception as e:
            last_exc = e
            if attempt == 0:
                logger.warning(
                    "Milvus search failed (%s); rebuilding client and retrying once...", e)
                _reset_client()
            else:
                raise
    else:
        raise last_exc  # 两次均失败

    hits = []
    for hit in (res[0] if res else []):
        entity = hit.get("entity", {}) if isinstance(hit, dict) else hit.entity
        distance = hit.get("distance") if isinstance(hit, dict) else hit.distance
        pk = hit.get("id") if isinstance(hit, dict) else hit.id
        name = entity.get(spec.image_name_field, "")
        hits.append({
            "key": name or str(pk),  # 去重键: 优先图片名
            "image_name": name,
            "text": entity.get(spec.text_field, "") or "",
            "similarity": float(distance) if distance is not None else None,
            "fields": {k: v for k, v in entity.items()},
        })
    return hits


def _rrf_merge(text_hits: list[dict], image_hits: list[dict],
               rrf_k: int = 60) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion 融合两路召回, 同 key 合并。"""
    merged: dict[str, dict[str, Any]] = {}
    for channel, hits in (("text", text_hits), ("image", image_hits)):
        for rank, h in enumerate(hits):
            m = merged.setdefault(h["key"], {
                "key": h["key"], "image_name": h["image_name"],
                "text": h["text"], "fields": h["fields"],
                "rrf_score": 0.0, "similarity": {}, "sources": [],
            })
            m["rrf_score"] += 1.0 / (rrf_k + rank + 1)
            m["similarity"][channel] = h["similarity"]
            m["sources"].append(channel)
            if not m["text"] and h["text"]:
                m["text"] = h["text"]
    return sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)


def preload_rag_models(dataset: Optional[str] = None) -> None:
    """在 agent 启动前预加载本工具用到的编码器并常驻显存。

    避免首次工具调用时现场加载: 现场加载会阻塞数分钟(PubMedBERT ~30s +
    MedImageInsight 读取 2.46GB 权重 ~2min), 且加载期间 Milvus 连接空转
    过久会被 gRPC 服务端以 too_many_pings 掐断。预加载后首次工具调用即秒回。
    重复调用是幂等的(编码器注册表为单例)。
    """
    spec = get_dataset(dataset)
    get_text_encoder(spec.text_encoder_key)
    get_image_encoder(spec.image_encoder_key)
    logger.info(
        "RAG encoders preloaded and resident: text=%s, image=%s",
        spec.text_encoder_key, spec.image_encoder_key,
    )


# ---------------------------------------------------------------------------
# 纯函数入口（流水线可直接调用, 不依赖 langchain）
# ---------------------------------------------------------------------------


def run_multimodal_rag_search(
    query: str,
    image_path: str,
    case_id: Optional[str] = None,
    dataset: Optional[str] = None,
    top_k: int = 5,
    max_images: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """双路检索: 关键词→文本向量, 图像→图像向量, RRF 融合后返回
    (content_blocks, result)。content_blocks 供多模态 LLM 阅读。"""

    def _error_return(msg: str, tb: Optional[str] = None):
        text = f"Multimodal RAG search failed: {msg}"
        if tb:
            text += f"\n\nFull traceback:\n{tb}"
        blocks = [{"type": "text", "text": text}]
        result = {
            "success": False, "dataset": dataset, "query": query,
            "case_id": case_id, "total_hits": 0, "n_text_hits": 0,
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

    # 2. 双路检索（异常完整暴露: 阶段标签 + 完整 traceback
    #    同时进入日志、LLM 可见的 content block 和 result["traceback"]）
    stage = "preparing filter"
    try:
        flt = _build_filter(spec, case_id, warnings)

        # FIFO 门闸: 多智能体多线程并发调用时, 同一时刻只允许一个请求
        # 做编码+检索, 其余按到达顺序排队, 防止并发把显存/内存打爆 (OOM)。
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
        logger.exception("Multimodal RAG retrieval failed at stage [%s]", stage)
        error_blocks, error_result = _error_return(
            f"retrieval error at [{stage}]: {type(e).__name__}: {e}", tb=tb)
        _notify_rag_observers(
            _rag_request_info(query, image_path, case_id, dataset, top_k),
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
        h["image_path"] = _resolve_image_path(spec, h["image_name"])
        if h["image_name"] and not h["image_path"]:
            warnings.append(f"Image file not found for '{h['image_name']}'")

    # 4. content_blocks: 概览 + 逐条"文本块 紧贴 图片块"交错布局。
    #    每条命中的描述块后面立即跟它的图片块, 大模型无需做全局顺序换算
    #    即可确定文本↔图片的对应关系(局部相邻, 不会错配)。
    overview = [
        f"## Multimodal RAG Results (dataset: {spec.key})\n",
        f"- Query keywords: {query}",
        f"- Query image: {image_path}",
        f"- case_id filter: {case_id or '(none)'}",
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
            f"- image: {h['image_name']}",
        ]
        if extra:
            hit_lines.append(f"- meta: {extra}")
        hit_lines.append(f"- description: {h['text'] or '(no description)'}")
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
        "case_id": case_id,
        "total_hits": len(fused),
        "n_text_hits": len(text_hits),
        "n_image_hits": len(image_hits),
        "hits": [
            {
                "rank": i,
                "dataset": spec.key,
                "image_name": h["image_name"],
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
        _rag_request_info(query, image_path, case_id, dataset, top_k),
        result,
    )
    return content_blocks, result


# ---------------------------------------------------------------------------
# LangChain tool
# ---------------------------------------------------------------------------


class MultimodalRagSearchInput(BaseModel):
    query: str = Field(
        description=(
            "Required. Search keywords describing the medical finding or "
            "anatomy of interest, e.g. 'pulmonary embolism filling defect' "
            "or 'PCL tear high signal'. Used for text-vector retrieval."
        ),
        min_length=1,
    )
    image_path: str = Field(
        description=(
            "Required. Absolute path to the query image file (PNG/JPG). "
            "Used for content-based image-vector retrieval against the "
            "knowledge base."
        ),
        min_length=1,
    )
    case_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional. Case identifier (e.g. 'MPX1009') to restrict "
            "retrieval to images of the same case."
        ),
    )
    dataset: Optional[str] = Field(
        default=None,
        description=(
            "Optional. Dataset key to search (default: the default "
            "collection). Available datasets can be listed from the tool "
            "module via list_datasets()."
        ),
    )
    top_k: Optional[int] = Field(
        default=5,
        description="Optional. Max number of fused results to return (default 5).",
    )


@tool(args_schema=MultimodalRagSearchInput)
def retrieve_medical_knowledge(
    query: str,
    image_path: str,
    case_id: Optional[str] = None,
    dataset: Optional[str] = None,
    top_k: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """从多模态医学影像知识库中检索参考资料（文本描述 + 图像）。

    Search the medical imaging knowledge base with BOTH text and image
    channels: the keywords are embedded and matched against stored report
    descriptions, while the query image is embedded and matched against
    stored images; the two result lists are fused (RRF) and the top hits
    are returned as text descriptions plus the actual reference images,
    which you can view directly.

    Inputs:
    - query: Search keywords for the finding/anatomy (required)
    - image_path: Path to the query image file (required)
    - case_id: Optional case id (e.g. 'MPX1009') to filter within one case
    - dataset: Optional dataset key (defaults to the default collection)
    - top_k: Optional max fused results (default 5)

    Outputs (content_blocks → visible to LLM):
    - Text: Ranked hits with description, source channel(s), similarities
    - Image(s): Reference images of the hits as data URLs

    Outputs (result dict → application metadata):
    - success / total_hits / n_text_hits / n_image_hits
    - hits: [{rank, dataset, image_name, image_path, text, sources,
      similarity, rrf_score, fields}]
    - warnings: Non-fatal issues (e.g. missing image files)
    - message: Status description
    """
    return run_multimodal_rag_search(
        query=query,
        image_path=image_path,
        case_id=case_id,
        dataset=dataset,
        top_k=top_k or 5,
    )
